from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
import numpy as np

threshold = 60*10

event_acc = EventAccumulator(your_tfevent_folder)
event_acc.Reload()
print(event_acc.Tags())

writer = SummaryWriter(output_folder)
for tag in event_acc.Tags()['scalars']:
    w_times, step_nums, vals = zip(*event_acc.Scalars(tag))
    start_time = w_times[0]
    w_times_diff = np.diff(w_times)
    # assert sum(w_times_diff > threshold) < 4
    for idx, val in enumerate(w_times_diff):
        if val > threshold:
            w_times_diff[idx] = (w_times_diff[idx-1]+w_times_diff[idx+1])/2
    w_times_merged = np.concatenate(([start_time], w_times_diff)).cumsum()
    for walltime, global_step, scalar_value in zip(w_times_merged, step_nums, vals):
        writer.add_scalar(tag=tag, walltime=walltime,
                        global_step=global_step, scalar_value=scalar_value)
writer.close()
print("Done")
