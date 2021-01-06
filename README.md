# tfevent_merger
Tensorboard calculates the "Relative" field by comparing the wallclock time of all data entries and the wallclock time of the first entry. Thus when you interrupt and restart the training, the "Relative" field includes the period that you are not actually running the training. I put together this helper script to trim the idle periods from the "Relative" field and make it show the actual time the training runs.

**WARNING: This script will change the wallclock time (in order to show correct relative time).**

To use it, simply change the `your_tfevent_folder`, `output_folder` and `threshold`(seconds). It only changes the sclars. To use on other fields, feel free to put loop over `event_acc.Tags()`.

#### Requirements
```
tensorboard
Pytorch
```

