package io.temporal.googleadk.activity;

import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;

/** The activity that runs a model call on a worker. Registered by the plugin. */
@ActivityInterface(namePrefix = "googleadk.")
public interface ModelActivity {

  /** The activity type name of {@link #invokeModel(InvokeModelInput)}. */
  String INVOKE_MODEL = "googleadk.InvokeModel";

  /** Calls the model named in the input and returns every response it produced. */
  @ActivityMethod
  InvokeModelOutput invokeModel(InvokeModelInput input);
}
