package io.temporal.googleadk.workflow;

import io.reactivex.rxjava3.core.Scheduler;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.schedulers.Schedulers;
import io.temporal.workflow.Async;
import io.temporal.workflow.Workflow;
import java.time.Duration;
import java.util.PriorityQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * An RxJava scheduler backed by Temporal workflow threads and timers, for ADK's internal fan-out.
 *
 * <p>ADK runs parallel agents and parallel tool calls on the scheduler carried by the invocation.
 * The {@linkplain #parallel() parallel} scheduler starts each task on its own workflow thread
 * ({@link Async#procedure}), so sibling agents and tools wait on their activities concurrently; the
 * {@linkplain #sequential() sequential} scheduler runs tasks one after another on the calling
 * thread. A delayed task waits on a workflow timer and runs in order of its due time and, for equal
 * due times, of its submission, as RxJava's time-based operators expect, and the scheduler and its
 * workers report workflow time as the current time, so periodic tasks keep their period in workflow
 * time. Both schedulers are deterministic and replay-safe.
 */
public final class WorkflowScheduler extends Scheduler {

  private final boolean parallel;

  private WorkflowScheduler(boolean parallel) {
    this.parallel = parallel;
  }

  /** Returns a scheduler that runs each task on its own workflow thread. */
  public static WorkflowScheduler parallel() {
    return new WorkflowScheduler(true);
  }

  /**
   * Returns a scheduler that runs tasks one at a time on the calling workflow thread. A delayed
   * task still waits and runs on a workflow thread of its own.
   */
  public static WorkflowScheduler sequential() {
    return new WorkflowScheduler(false);
  }

  @Override
  public Worker createWorker() {
    return new WorkflowWorker(parallel);
  }

  @Override
  public Disposable scheduleDirect(Runnable run, long delay, TimeUnit unit) {
    return createWorker().schedule(run, delay, unit);
  }

  @Override
  public long now(TimeUnit unit) {
    return unit.convert(Workflow.currentTimeMillis(), TimeUnit.MILLISECONDS);
  }

  private static final class WorkflowWorker extends Worker {
    private final boolean parallel;
    private final Worker trampoline;
    private final AtomicBoolean disposed = new AtomicBoolean();
    // Delayed tasks, ordered by due time then submission; drained by whichever sleeper wakes first.
    private final PriorityQueue<DelayedTask> delayed = new PriorityQueue<>();
    private long submissions;

    WorkflowWorker(boolean parallel) {
      this.parallel = parallel;
      this.trampoline = parallel ? null : Schedulers.trampoline().createWorker();
    }

    @Override
    public Disposable schedule(Runnable run, long delay, TimeUnit unit) {
      if (disposed.get()) {
        return Disposable.disposed();
      }
      Task task = new Task(run, disposed);
      if (delay <= 0) {
        start(task);
        return task;
      }
      Duration pause = Duration.ofMillis(unit.toMillis(delay));
      DelayedTask entry =
          new DelayedTask(Workflow.currentTimeMillis() + pause.toMillis(), submissions++, task);
      delayed.add(entry);
      Async.procedure(
          () -> {
            Workflow.sleep(pause);
            drainDue();
            // The sleeper's own task runs even if the clock reads earlier than its due time, after
            // whatever is queued ahead of it, so submission order holds for equal due times.
            while (!delayed.isEmpty() && delayed.peek().compareTo(entry) <= 0) {
              delayed.poll().task.run();
            }
          });
      return task;
    }

    /** Runs every delayed task that is due, in order; later sleepers find nothing left. */
    private void drainDue() {
      long now = Workflow.currentTimeMillis();
      while (!delayed.isEmpty() && delayed.peek().due <= now) {
        delayed.poll().task.run();
      }
    }

    private void start(Task task) {
      if (parallel) {
        Async.procedure(task::run);
      } else {
        trampoline.schedule(task);
      }
    }

    @Override
    public void dispose() {
      if (disposed.compareAndSet(false, true)) {
        delayed.clear();
        if (trampoline != null) {
          trampoline.dispose();
        }
      }
    }

    @Override
    public boolean isDisposed() {
      return disposed.get();
    }

    /** Returns workflow time, which RxJava's periodic tasks use to place their next tick. */
    @Override
    public long now(TimeUnit unit) {
      return unit.convert(Workflow.currentTimeMillis(), TimeUnit.MILLISECONDS);
    }
  }

  private static final class DelayedTask implements Comparable<DelayedTask> {
    final long due;
    final long sequence;
    final Task task;

    DelayedTask(long due, long sequence, Task task) {
      this.due = due;
      this.sequence = sequence;
      this.task = task;
    }

    @Override
    public int compareTo(DelayedTask other) {
      int byDue = Long.compare(due, other.due);
      return byDue != 0 ? byDue : Long.compare(sequence, other.sequence);
    }
  }

  private static final class Task implements Runnable, Disposable {
    private final Runnable run;
    private final AtomicBoolean workerDisposed;
    private final AtomicBoolean disposed = new AtomicBoolean();

    Task(Runnable run, AtomicBoolean workerDisposed) {
      this.run = run;
      this.workerDisposed = workerDisposed;
    }

    @Override
    public void run() {
      if (!disposed.get() && !workerDisposed.get()) {
        run.run();
      }
    }

    @Override
    public void dispose() {
      disposed.set(true);
    }

    @Override
    public boolean isDisposed() {
      return disposed.get();
    }
  }
}
