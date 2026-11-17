import test from 'ava';
import { mergeModelOptions, DEFAULT_MODEL_OPTIONS, type ModelSummaryProvider } from '../common/model-options';

test('mergeModelOptions: header only — per-client fields win over DEFAULT', (t) => {
  const merged = mergeModelOptions(undefined, {
    activityOptions: { startToCloseTimeout: '99s' },
    useLocalActivity: true,
  });
  t.is(merged.activityOptions.startToCloseTimeout, '99s');
  t.is(merged.useLocalActivity, true);
});

test('mergeModelOptions: neither header nor constructor — returns DEFAULT', (t) => {
  const merged = mergeModelOptions(undefined, undefined);
  t.is(merged.activityOptions.startToCloseTimeout, DEFAULT_MODEL_OPTIONS.activityOptions.startToCloseTimeout);
  t.is(merged.useLocalActivity, DEFAULT_MODEL_OPTIONS.useLocalActivity);
});

test('mergeModelOptions: constructor only, no header — header-less Workflow gets activityOptions', (t) => {
  // A Schedule/UI/CLI-started Workflow has no config header, so the constructor
  // default is the only source layered above DEFAULT.
  const merged = mergeModelOptions(
    { activityOptions: { startToCloseTimeout: '120s', heartbeatTimeout: 0 }, streamingTopic: 'events' },
    undefined
  );
  t.is(merged.activityOptions.startToCloseTimeout, '120s');
  t.is(merged.streamingTopic, 'events');
  t.is(merged.useLocalActivity, false);
  t.is(merged.activityOptions.heartbeatTimeout, 0);
});

test('mergeModelOptions: both present — client overrides matching field, non-overlapping constructor fields survive', (t) => {
  // Merge-trap regression: the header layer must carry ONLY client-set fields. If
  // DEFAULT were pre-baked into it, DEFAULT's startToCloseTimeout/useLocalActivity
  // would clobber the constructor's even though the client never set them.
  const merged = mergeModelOptions(
    { activityOptions: { startToCloseTimeout: '120s' }, useLocalActivity: true, streamingTopic: 'events' },
    { activityOptions: { startToCloseTimeout: '30s' } }
  );
  t.is(merged.activityOptions.startToCloseTimeout, '30s');
  t.is(merged.streamingTopic, 'events');
  t.is(merged.useLocalActivity, true);
});

test('mergeModelOptions: streamingTopic supplied only by the runner', (t) => {
  const merged = mergeModelOptions({ streamingTopic: 'events' }, { activityOptions: { startToCloseTimeout: '30s' } });
  t.is(merged.streamingTopic, 'events');
});

test('mergeModelOptions: an explicit undefined field never clobbers a lower layer', (t) => {
  const merged = mergeModelOptions(
    { streamingTopic: 'events' },
    {
      streamingTopic: undefined,
      activityOptions: { startToCloseTimeout: '30s', summary: undefined },
    }
  );
  t.is(merged.streamingTopic, 'events');
});

test('mergeModelOptions: a defined falsy field in the header overrides a truthy lower layer', (t) => {
  // Guards merging against a truthiness check: `false` is defined and must
  // win over the constructor's `true`, not be dropped as if it were unset.
  const merged = mergeModelOptions({ useLocalActivity: true }, { useLocalActivity: false });
  t.is(merged.useLocalActivity, false);
});

test('mergeModelOptions: a defined 0 in the header overrides a lower layer', (t) => {
  const merged = mergeModelOptions({ streamingBatchInterval: 100 }, { streamingBatchInterval: 0, activityOptions: { heartbeatTimeout: 0 } });
  t.is(merged.streamingBatchInterval, 0);
  t.is(merged.activityOptions.heartbeatTimeout, 0);
});

test('mergeModelOptions: Workflow-authored summaryProvider passes through', (t) => {
  const provider: ModelSummaryProvider = { provide: () => 'dynamic' };
  const merged = mergeModelOptions({ summaryProvider: provider }, undefined);
  t.is(merged.summaryProvider, provider);
});
