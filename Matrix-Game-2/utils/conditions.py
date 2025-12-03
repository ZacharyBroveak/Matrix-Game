
import torch
import random

def combine_data(data, num_frames=57, keyboard_dim=6, mouse=True):
    assert num_frames % 4 == 1
    keyboard_condition = torch.zeros((num_frames, keyboard_dim))
    if mouse == True:
        mouse_condition = torch.zeros((num_frames, 2))
    
    current_frame = 0
    selections = [12]

    while current_frame < num_frames:
        rd_frame = selections[random.randint(0, len(selections) - 1)]
        rd = random.randint(0, len(data) - 1)
        k = data[rd]['keyboard_condition']
        if mouse == True:
            m = data[rd]['mouse_condition']
        
        if current_frame == 0:
            keyboard_condition[:1] = k[:1]
            if mouse == True:
                mouse_condition[:1] = m[:1]
            current_frame = 1
        else:
            rd_frame = min(rd_frame, num_frames - current_frame)
            repeat_time = rd_frame // 4
            keyboard_condition[current_frame:current_frame+rd_frame] = k.repeat(repeat_time, 1)
            if mouse == True:
                mouse_condition[current_frame:current_frame+rd_frame] = m.repeat(repeat_time, 1)
            current_frame += rd_frame
    if mouse == True:
        return {
                "keyboard_condition": keyboard_condition,
                "mouse_condition": mouse_condition
            }
    return {"keyboard_condition": keyboard_condition}

def Bench_actions_universal(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        # "back",
        "left",
        "right",
    ]
    actions_double_action = [
        "forward_left",
        "forward_right",
        # "back_left",
        # "back_right",
    ]

    actions_single_camera = [   
        "camera_l",
        "camera_r",
        # "camera_ur",
        # "camera_ul",
        # "camera_dl",
        # "camera_dr" 
        # "camera_up",
        # "camera_down",
    ]
    actions_to_test = actions_double_action * 5 + actions_single_camera * 5 + actions_single_action * 5
    for action in (actions_single_action + actions_double_action):
        for camera in (actions_single_camera):
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    # print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1, "left": 2, "right": 3
    }

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_up":  [CAM_VALUE, 0],
        "camera_down": [-CAM_VALUE, 0],
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
        "camera_ur":  [CAM_VALUE, CAM_VALUE],
        "camera_ul":  [CAM_VALUE, -CAM_VALUE],
        "camera_dr":  [-CAM_VALUE, CAM_VALUE],
        "camera_dl":  [-CAM_VALUE, -CAM_VALUE],
    }

    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0, 0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=4, mouse=True)

"""
def Bench_actions_gta_drive(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        "back",
    ]

    actions_single_camera = [   
        "camera_l",
        "camera_r",
    ]
    actions_to_test = actions_single_camera * 2 + actions_single_action * 2
    for action in (actions_single_action):
        for camera in (actions_single_camera):
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    # print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1
    }

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
    }
    
    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=2, mouse=True)
"""

def Bench_actions_gta_drive(num_frames,
                            num_samples_per_action=4,        # kept for backward compat
                            script=None,                      # e.g., ["forward","forward","camera_l","forward","back",...]
                            frames_per_token=4,            # e.g., 4; defaults to num_samples_per_action
                            cam_mag=0.10):                    # steer/camera magnitude
    KEYBOARD_IDX = {"forward": 0, "back": 1}
    CAMERA_VALUE_MAP = {"camera_l": [0, -cam_mag], "camera_r": [0, cam_mag]}
    ALLOWED_TOKENS = set(KEYBOARD_IDX) | set(CAMERA_VALUE_MAP) | {"neutral", "none", ""}

    def _tokenize(entry):
        """Split a script entry into primitive actions, preserving order."""
        if isinstance(entry, (list, tuple)):
            tokens = []
            for part in entry:
                tokens.extend(_tokenize(part))
            return tokens
        if not isinstance(entry, str):
            raise TypeError(f"Unsupported script entry type: {type(entry)}")

        s = entry.strip().lower()
        if s in {"neutral", "none", ""}:
            return []
        if s in ALLOWED_TOKENS:
            return [s]

        tokens = []
        pos = 0
        # Match longest tokens first so we don't split camera_l into camera + l.
        token_candidates = sorted(ALLOWED_TOKENS - {"neutral", "none", ""}, key=len, reverse=True)
        while pos < len(s):
            if s[pos] in "_+ ,":
                pos += 1
                continue
            matched = False
            for candidate in token_candidates:
                if s.startswith(candidate, pos):
                    tokens.append(candidate)
                    pos += len(candidate)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Unknown action token '{s}' in script entry '{entry}'")
        return tokens

    # --- Deterministic path ---
    if script is not None and len(script) > 0:
        fpt = frames_per_token or num_samples_per_action
        data = []
        for act in script:
            tokens = _tokenize(act)
            kb = [[0, 0] for _ in range(fpt)]
            ms = [[0, 0] for _ in range(fpt)]
            mouse_dx = mouse_dy = 0.0

            for token in tokens:
                if token in KEYBOARD_IDX:
                    col = KEYBOARD_IDX[token]
                    for row in kb:
                        row[col] = 1
                if token in CAMERA_VALUE_MAP:
                    dx, dy = CAMERA_VALUE_MAP[token]
                    mouse_dx += dx
                    mouse_dy += dy

            if mouse_dx != 0.0 or mouse_dy != 0.0:
                ms = [[mouse_dx, mouse_dy] for _ in range(fpt)]
            data.append({"keyboard_condition": torch.tensor(kb),
                         "mouse_condition": torch.tensor(ms)})
        # Build a deterministic, ordered sequence from the script and tile it
        # to fill num_frames. Each script entry currently has fpt rows; we
        # concatenate them in order and repeat the full sequence until we
        # reach num_frames. This preserves the order and ensures each action
        # lasts `frames_per_token` frames.
        seq_k = torch.cat([d["keyboard_condition"] for d in data], dim=0) if len(data) > 0 else torch.zeros((0, 2))
        seq_m = torch.cat([d["mouse_condition"] for d in data], dim=0) if len(data) > 0 else torch.zeros((0, 2))
        total_len = seq_k.shape[0]
        if total_len == 0:
            # Fallback to the original combine behavior if nothing was built
            return combine_data(data, num_frames, keyboard_dim=2, mouse=True)

        repeats = (num_frames + total_len - 1) // total_len
        keyboard_condition = seq_k.repeat(repeats, 1)[:num_frames]
        mouse_condition = seq_m.repeat(repeats, 1)[:num_frames]

        return {
            "keyboard_condition": keyboard_condition,
            "mouse_condition": mouse_condition
        }

    # --- Original stochastic/batch path (unchanged) ---
    actions_single_action = ["forward", "back"]
    actions_single_camera = ["camera_l","camera_r"]
    actions_to_test = actions_single_camera * 2 + actions_single_action * 2
    for action in (actions_single_action):
        for camera in (actions_single_camera):
            actions_to_test.append(f"{action}_{camera}")
    base_action = actions_single_action + actions_single_camera

    data = []
    for action_name in actions_to_test:
        keyboard_condition = [[0, 0] for _ in range(num_samples_per_action)]
        mouse_condition = [[0, 0] for _ in range(num_samples_per_action)]
        for sub_act in base_action:
            if sub_act not in action_name:
                continue
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act] for _ in range(num_samples_per_action)]
            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1
        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=2, mouse=True)

def Bench_actions_templerun(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "jump",
        "slide",
        "leftside",
        "rightside",
        "turnleft",
        "turnright",
        "nomove"
    ]

    actions_to_test = actions_single_action

    base_action = actions_single_action

    KEYBOARD_IDX = { 
        "nomove": 0, "jump": 1, "slide": 2, "turnleft": 3,
        "turnright": 4, "leftside": 5, "rightside": 6
    }

    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=7, mouse=False)
