using System.Linq.Expressions;
using System.Reflection;
using Microsoft.Extensions.AI;
using Temporalio.Extensions.MicrosoftAgentFramework.Internal;
using Temporalio.Workflows;

namespace Temporalio.Extensions.MicrosoftAgentFramework;

/// <summary>
/// Wraps one of your Temporal Activities as an <see cref="AIFunction"/>. The schema comes from
/// the method's parameters; each invocation is one Activity run with the options you pass.
/// Register the Activity on the worker as usual.
/// </summary>
public static class ActivityAsTool
{
    /// <summary>
    /// Wraps an instance Activity named through a method-call expression, the way
    /// <c>Workflow.ExecuteActivityAsync</c> names one. Argument values in the expression are
    /// ignored; the model supplies them.
    /// </summary>
    /// <typeparam name="TActivityInstance">Class that holds the Activity.</typeparam>
    /// <typeparam name="TResult">Return type of the call in the expression.</typeparam>
    /// <param name="activityCall">For example <c>(MyActivities a) => a.GetWeather(default!)</c>.</param>
    /// <param name="options">Activity options; a start-to-close or schedule-to-close timeout is required.</param>
    /// <param name="name">Tool name the model sees; defaults to the Activity name.</param>
    /// <param name="description">Tool description; defaults to the method's <see cref="System.ComponentModel.DescriptionAttribute"/>.</param>
    /// <returns>The tool.</returns>
    public static AIFunction Create<TActivityInstance, TResult>(
        Expression<Func<TActivityInstance, TResult>> activityCall,
        ActivityOptions options,
        string? name = null,
        string? description = null) =>
        Create(ExtractMethod(activityCall), options, name, description);

    /// <summary>
    /// Wraps a static Activity named through a method-call expression.
    /// </summary>
    /// <typeparam name="TResult">Return type of the call in the expression.</typeparam>
    /// <param name="activityCall">For example <c>() => MyActivities.Fetch(default!)</c>.</param>
    /// <param name="options">Activity options; a start-to-close or schedule-to-close timeout is required.</param>
    /// <param name="name">Tool name the model sees; defaults to the Activity name.</param>
    /// <param name="description">Tool description; defaults to the method's <see cref="System.ComponentModel.DescriptionAttribute"/>.</param>
    /// <returns>The tool.</returns>
    public static AIFunction Create<TResult>(
        Expression<Func<TResult>> activityCall,
        ActivityOptions options,
        string? name = null,
        string? description = null) =>
        Create(ExtractMethod(activityCall), options, name, description);

    /// <summary>
    /// Wraps an Activity given as a delegate, such as a method group.
    /// </summary>
    /// <param name="activity">Delegate whose method carries <see cref="Temporalio.Activities.ActivityAttribute"/>.</param>
    /// <param name="options">Activity options; a start-to-close or schedule-to-close timeout is required.</param>
    /// <param name="name">Tool name the model sees; defaults to the Activity name.</param>
    /// <param name="description">Tool description; defaults to the method's <see cref="System.ComponentModel.DescriptionAttribute"/>.</param>
    /// <returns>The tool.</returns>
    public static AIFunction Create(Delegate activity, ActivityOptions options, string? name = null, string? description = null)
    {
        ArgumentNullException.ThrowIfNull(activity);
        return Create(activity.Method, options, name, description);
    }

    /// <summary>
    /// Wraps an Activity given as a <see cref="MethodInfo"/>.
    /// </summary>
    /// <param name="activity">Method that carries <see cref="Temporalio.Activities.ActivityAttribute"/>.</param>
    /// <param name="options">Activity options; a start-to-close or schedule-to-close timeout is required.</param>
    /// <param name="name">Tool name the model sees; defaults to the Activity name.</param>
    /// <param name="description">Tool description; defaults to the method's <see cref="System.ComponentModel.DescriptionAttribute"/>.</param>
    /// <returns>The tool.</returns>
    public static AIFunction Create(MethodInfo activity, ActivityOptions options, string? name = null, string? description = null) =>
        new ActivityFunction(activity, options, name, description);

    private static MethodInfo ExtractMethod(LambdaExpression expression)
    {
        ArgumentNullException.ThrowIfNull(expression);
        var body = expression.Body;
        while (body is UnaryExpression { NodeType: ExpressionType.Convert } convert)
        {
            body = convert.Operand;
        }

        if (body is not MethodCallExpression call)
        {
            throw new ArgumentException(
                "The expression must be a single call to a method that carries [Activity], for example (MyActivities a) => a.GetWeather(default!). " +
                "To wrap a lambda body, make it an Activity method first.",
                nameof(expression));
        }

        return call.Method;
    }
}
