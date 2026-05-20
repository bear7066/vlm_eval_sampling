This is the v1.0 release of the AVA-Kinetics action localization dataset.

It consists of the AVA Atomic Visual Actions dataset (version 2.2) combined with
frames from the Kinetics-700 dataset, annotated with person bounding boxes as
well as the localized actions from the AVA vocabulary.

===

DATA FORMAT:

Each row of a CSV file contains an annotation for one person performing an
action in an interval, where that annotation is associated with the middle
keyframe. Different persons and multiple action labels are described in separate
rows.

The format of a row is the following: video_id, middle_frame_timestamp,
person_box, action_id, person_id

video_id: YouTube identifier
keyframe_timestamp: in seconds from the start of the YouTube video
person_box: top-left (x1, y1) and bottom-right (x2,y2) normalized with respect
  to frame size, where (0.0, 0.0) corresponds to the top left, and (1.0, 1.0)
  corresponds to the bottom right.
action_id: identifier of an action class, see ava_action_list_v2.2.pbtxt
person_id: a unique integer allowing this box to be linked to other boxes
  depicting the same person in adjacent frames of this video. The person id is
  optional--included in the AVA data but not in the Kinetics data.

In the training and validation files, rows containing only a
video_id,keyframe_timestamp (i.e. no box or action label) are included to
indicate an empty frame where no action was identified by labelers.

In the test files, boxes and action labels are omitted entirely, rather the
video_id,keyframe_timestamp pairs indicate the video frames upon which a
submission will be tested.

The full action label vocabulary is provided in
ava_action_list_v2.2_for_activitynet.pbtxt.

For the ActivityNet challenge, and in many papers, results are reported on only
a subset of 60 actions listed in ava_action_list_v2.2_for_activitynet.pbtxt.

===

EVALUATION:

Evaluation code used for the ActivityNet Challenge may be found at https://github.com/activitynet/ActivityNet/blob/master/Evaluation/get_ava_performance.py

The ground truth labels on the test set have not been released to the public. To evaluate on the test set, a submission must be made to the ActivityNet Evaluation Server, http://activity-net.org/challenges/2020/evaluation.html

===

For more about AVA-Kinetics or AVA, please visit
https://research.google.com/ava/.

For more about Kinetics, please visit
https://deepmind.com/research/open-source/kinetics.

Please send any questions to the Google Group ava-dataset-users:
https://groups.google.com/forum/#!forum/ava-dataset-users


