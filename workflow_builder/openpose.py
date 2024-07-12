def convert_katalist_to_openpose(keypoints, width, height) -> dict:
    """Convert the pose keypoints from the Katalist format to the OpenPose format."""
    pose_keypoint = {
        "people": [],
        "canvas_width": width,
        "canvas_height": height,
    }
    for person in keypoints:
        pose_keypoints_2d = []
        for point in person:
            pose_keypoints_2d.append(point["x"] * width)
            pose_keypoints_2d.append(point["y"] * height)
            pose_keypoints_2d.append(1.0)
        pose_keypoint["people"].append(
            {
                "pose_keypoints_2d": pose_keypoints_2d,
                "face_keypoints_2d": None,
                "hand_left_keypoints_2d": None,
                "hand_right_keypoints_2d": None,
            }
        )
    return pose_keypoint
