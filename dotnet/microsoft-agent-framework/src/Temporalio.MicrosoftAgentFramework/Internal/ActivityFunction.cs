using System.ComponentModel;
using System.Reflection;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Temporalio.Activities;
using Temporalio.Workflows;

namespace Temporalio.MicrosoftAgentFramework.Internal;

/// <summary>
/// An <see cref="AIFunction"/> whose invocation is one Temporal Activity. Built by
/// <see cref="ActivityAsTool"/>.
/// </summary>
internal sealed class ActivityFunction : AIFunction
{
    private static readonly MethodInfo RunWithResultMethod = typeof(ActivityFunction)
        .GetMethod(nameof(RunWithResultAsync), BindingFlags.NonPublic | BindingFlags.Static)!;

    private readonly string activityName;
    private readonly ActivityOptions options;
    private readonly ParameterInfo[] parameters;
    private readonly int argumentCount;
    private readonly Type? resultType;
    private readonly MethodInfo? runWithResult;

    /// <summary>
    /// Initializes a new instance of the <see cref="ActivityFunction"/> class.
    /// </summary>
    /// <param name="method">The Activity method; it must carry <see cref="ActivityAttribute"/>.</param>
    /// <param name="options">Activity options; a start-to-close or schedule-to-close timeout is required.</param>
    /// <param name="name">Tool name, or null for the Activity name.</param>
    /// <param name="description">Tool description, or null for the method's <see cref="DescriptionAttribute"/>.</param>
    public ActivityFunction(MethodInfo method, ActivityOptions options, string? name, string? description)
    {
        ArgumentNullException.ThrowIfNull(method);
        ArgumentNullException.ThrowIfNull(options);
        if (options.StartToCloseTimeout is null && options.ScheduleToCloseTimeout is null)
        {
            throw new ArgumentException("ActivityOptions needs StartToCloseTimeout or ScheduleToCloseTimeout.", nameof(options));
        }

        var definition = ActivityDefinition.Create(method, _ => null);
        activityName = definition.Name ?? throw new ArgumentException("Dynamic activities cannot be tools.", nameof(method));
        Name = name ?? activityName;
        Description = description ?? method.GetCustomAttribute<DescriptionAttribute>()?.Description ?? string.Empty;
        UnderlyingMethod = method;
        parameters = method.GetParameters();
        argumentCount = CountArguments(parameters, Name);
        JsonSchema = AIJsonUtilities.CreateFunctionJsonSchema(method, title: null, description: Description, AIJsonUtilities.DefaultOptions);
        this.options = (ActivityOptions)options.Clone();
        this.options.Summary ??= Name;
        resultType = UnwrapResultType(method.ReturnType);
        runWithResult = resultType is null ? null : RunWithResultMethod.MakeGenericMethod(resultType);
    }

    /// <inheritdoc />
    public override string Name { get; }

    /// <inheritdoc />
    public override string Description { get; }

    /// <inheritdoc />
    public override JsonElement JsonSchema { get; }

    /// <inheritdoc />
    public override MethodInfo UnderlyingMethod { get; }

    /// <summary>Gets the Activity name the tool schedules.</summary>
    public string ActivityName => activityName;

    /// <inheritdoc />
    protected override async ValueTask<object?> InvokeCoreAsync(AIFunctionArguments arguments, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(arguments);
        var args = BindArguments(arguments);
        var activityOptions = options;
        if (cancellationToken.CanBeCanceled && activityOptions.CancellationToken is null)
        {
            activityOptions = (ActivityOptions)activityOptions.Clone();
            activityOptions.CancellationToken = cancellationToken;
        }

        if (runWithResult is null)
        {
            await Workflow.ExecuteActivityAsync(activityName, args, activityOptions).ConfigureAwait(true);
            return null;
        }

        var task = (Task<object?>)runWithResult.Invoke(null, [activityName, args, activityOptions])!;
        return await task.ConfigureAwait(true);
    }

    private static async Task<object?> RunWithResultAsync<TResult>(string activity, object?[] args, ActivityOptions options) =>
        await Workflow.ExecuteActivityAsync<TResult>(activity, args, options).ConfigureAwait(true);

    private static int CountArguments(ParameterInfo[] parameters, string name)
    {
        // A CancellationToken cannot cross the Activity boundary. The SDK fills trailing optional
        // parameters that get no argument, so trailing token parameters with a default value are
        // left out of the call; the Activity reads cancellation from ActivityExecutionContext.Current.
        var count = parameters.Length;
        while (count > 0 && parameters[count - 1].ParameterType == typeof(CancellationToken) && parameters[count - 1].HasDefaultValue)
        {
            count--;
        }

        if (Array.FindIndex(parameters, 0, count, parameter => parameter.ParameterType == typeof(CancellationToken)) >= 0)
        {
            throw new ArgumentException(
                $"Tool '{name}' wraps an Activity with a CancellationToken parameter that is not a trailing optional one. A token cannot cross the Activity boundary: " +
                "declare it last with a default value, or read ActivityExecutionContext.Current.CancellationToken inside the Activity.");
        }

        return count;
    }

    private static Type? UnwrapResultType(Type returnType)
    {
        if (returnType == typeof(void) || returnType == typeof(Task) || returnType == typeof(ValueTask))
        {
            return null;
        }

        if (returnType.IsGenericType)
        {
            var definition = returnType.GetGenericTypeDefinition();
            if (definition == typeof(Task<>) || definition == typeof(ValueTask<>))
            {
                return returnType.GetGenericArguments()[0];
            }
        }

        return returnType;
    }

    private static bool TryGetArgument(AIFunctionArguments arguments, string name, out object? value)
    {
        if (arguments.TryGetValue(name, out value))
        {
            return true;
        }

        foreach (var pair in arguments)
        {
            if (string.Equals(pair.Key, name, StringComparison.OrdinalIgnoreCase))
            {
                value = pair.Value;
                return true;
            }
        }

        value = null;
        return false;
    }

    private static object? Convert(object? value, Type targetType)
    {
        switch (value)
        {
            case null:
                return null;
            case JsonElement element:
                return element.Deserialize(targetType, AIJsonUtilities.DefaultOptions);
            default:
                if (targetType.IsInstanceOfType(value))
                {
                    return value;
                }

                var json = JsonSerializer.SerializeToElement(value, AIJsonUtilities.DefaultOptions);
                return json.Deserialize(targetType, AIJsonUtilities.DefaultOptions);
        }
    }

    private object?[] BindArguments(AIFunctionArguments arguments)
    {
        var args = new object?[argumentCount];
        for (var i = 0; i < argumentCount; i++)
        {
            var parameter = parameters[i];
            if (TryGetArgument(arguments, parameter.Name!, out var value))
            {
                args[i] = Convert(value, parameter.ParameterType);
            }
            else if (parameter.HasDefaultValue)
            {
                args[i] = parameter.DefaultValue;
            }
            else if (!parameter.ParameterType.IsValueType || Nullable.GetUnderlyingType(parameter.ParameterType) is not null)
            {
                args[i] = null;
            }
            else
            {
                throw new ArgumentException($"Tool '{Name}' was called without the required argument '{parameter.Name}'.");
            }
        }

        return args;
    }
}
