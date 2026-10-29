import torch
import random


def create_target_mask(mask_type,
                       embedding_type,
                       n_target_data,
                       n_target_theta,
                       n_selected_targets=None,
                       predefined_masks=None,
                       predefined_mask_weights=None,
                       mask_index=None,
                       attend_to=None):
    """Bool mask [n_target_data + n_target_theta], True = attend."""

    n_target = n_target_data + n_target_theta

    mask = torch.zeros(n_target, dtype=torch.bool)

    if mask_type == 'all':
        mask.fill_(True)

    elif mask_type == 'none':
        mask.fill_(False)

    elif mask_type == 'partial':
        if n_selected_targets is None:
            # default half
            n_selected_targets = max(1, n_target // 2)

        if embedding_type in ['data', 'theta', 'mix']:
            indices = torch.randperm(n_target)[:n_selected_targets]
            mask[indices] = True
        else:
            raise ValueError(f"Unknown embedding_type '{embedding_type}' for partial mask")

    elif mask_type == 'predefined':
        if predefined_masks is None or len(predefined_masks) == 0:
            raise ValueError("predefined_masks must be provided for mask_type='predefined'")

        if mask_index is not None:
            predefined_mask = predefined_masks[mask_index]
        else:
            if predefined_mask_weights is not None and len(predefined_mask_weights) == len(predefined_masks):
                weights = torch.tensor(predefined_mask_weights, dtype=torch.float)
                probabilities = weights / weights.sum()
                index = torch.multinomial(probabilities, 1).item()
                predefined_mask = predefined_masks[index]
            else:
                predefined_mask = random.choice(predefined_masks)

        for i, should_attend in enumerate(predefined_mask):
            if i < n_target and should_attend:
                mask[i] = True

    elif mask_type == 'split':
        if embedding_type != 'mix':
            raise ValueError("mask_type='split' only works with embedding_type='mix'")

        attend_to_data = True if attend_to == 'data' else False
        if attend_to is None:
            attend_to_data = random.choice([True, False])

        if attend_to_data:
            mask[:n_target_data] = True
        else:
            mask[n_target_data:] = True

    else:
        raise ValueError(f"Unknown mask_type '{mask_type}'")

    return mask


def select_targets_by_mask(target_results, target_mask):
    """Indexes target_results ([B,N,...]) down to the True entries of target_mask ([N])."""
    selected_indices = torch.where(target_mask)[0]

    selected_results = target_results[:, selected_indices]

    return selected_results


def get_masking_description(mask_type, embedding_type, n_selected_targets=None,
                             mask_index=None, attend_to=None):
    """
    Readable mask config, for logging.

    mask_type: 'all', 'none', 'partial', 'predefined', 'split'
    embedding_type: 'data', 'theta', 'mix'
    n_selected_targets: 'partial' only, mask_index: 'predefined' only, attend_to: 'split' only
    """
    if mask_type == 'all':
        return "Attending to all targets"
    elif mask_type == 'none':
        return "Not attending to any targets"
    elif mask_type == 'partial':
        return f"Attending to {n_selected_targets} randomly selected targets"
    elif mask_type == 'predefined':
        if mask_index is not None:
            return f"Using predefined mask #{mask_index}"
        else:
            return "Using randomly selected predefined mask"
    elif mask_type == 'split' and embedding_type == 'mix':
        if attend_to is not None:
            return f"Attending to all {attend_to} targets"
        else:
            return "Attending to either all data or all theta targets (random choice)"
    return "Unknown masking configuration"


def test_create_target_mask():
    """Test the create_target_mask function with various configurations."""
    from attrdictionary import AttrDict

    test_configs = [
        {"name": "All targets (data mode)",
         "cfg": AttrDict({"embedding_type": "data",
                          "task": AttrDict({"mask_type": "all", "n_target_data": 5, "n_target_theta": 0})}),
         "expected_mask": torch.ones(5, dtype=torch.bool),
         "expected_desc": "Attending to all targets"},

        {"name": "All targets (theta mode)",
         "cfg": AttrDict({"embedding_type": "theta",
                          "task": AttrDict({"mask_type": "all", "n_target_data": 0, "n_target_theta": 4})}),
         "expected_mask": torch.ones(4, dtype=torch.bool),
         "expected_desc": "Attending to all targets"},

        {"name": "All targets (mix mode)",
         "cfg": AttrDict({"embedding_type": "mix",
                          "task": AttrDict({"mask_type": "all", "n_target_data": 3, "n_target_theta": 2})}),
         "expected_mask": torch.ones(5, dtype=torch.bool),
         "expected_desc": "Attending to all targets"},

        {"name": "No targets",
         "cfg": AttrDict({"embedding_type": "data",
                          "task": AttrDict({"mask_type": "none", "n_target_data": 5, "n_target_theta": 0})}),
         "expected_mask": torch.zeros(5, dtype=torch.bool),
         "expected_desc": "Not attending to any targets"},

        {"name": "Partial targets",
         "cfg": AttrDict({"embedding_type": "data",
                          "task": AttrDict({"mask_type": "partial", "n_target_data": 5, "n_target_theta": 0, "n_selected_targets": 2,})}),
         "expected_count": 2,  # random indices, count only
         "expected_desc": "Attending to 2 randomly selected targets"},

        {"name": "Predefined mask",
         "cfg": AttrDict({"embedding_type": "theta",
                          "task": AttrDict({"predefined_masks": [[True, False, True, False]], "mask_type": "predefined", "n_target_data": 0, "n_target_theta": 4, "mask_index": 0})}),
         "expected_mask": torch.tensor([True, False, True, False]),
         "expected_desc": "Using predefined mask #0"},

        {"name": "Split mask (data)",
         "cfg": AttrDict({"embedding_type": "mix",
                          "task": AttrDict({"mask_type": "split", "n_target_data": 3, "n_target_theta": 2, "attend_to": "data",})}),
         "expected_mask": torch.tensor([True, True, True, False, False]),
         "expected_desc": "Attending to all data targets"},

        {"name": "Split mask (theta)",
         "cfg": AttrDict({"embedding_type": "mix",
                          "task": AttrDict({"mask_type": "split", "n_target_data": 3, "n_target_theta": 2, "attend_to": "theta",})}),
         "expected_mask": torch.tensor([False, False, False, True, True]),
         "expected_desc": "Attending to all theta targets"},
    ]

    for test in test_configs:
        print(f"\nTesting: {test['name']}")
        cfg, task = test["cfg"], test["cfg"].task
        mask = create_target_mask(
            task.mask_type,
            cfg.embedding_type,
            task.n_target_data,
            task.n_target_theta,
            n_selected_targets=getattr(task, "n_selected_targets", None),
            predefined_masks=getattr(task, "predefined_masks", None),
            predefined_mask_weights=getattr(task, "predefined_mask_weights", None),
            mask_index=getattr(task, "mask_index", None),
            attend_to=getattr(task, "attend_to", None),
        )
        desc = get_masking_description(
            task.mask_type,
            cfg.embedding_type,
            n_selected_targets=getattr(task, "n_selected_targets", None),
            mask_index=getattr(task, "mask_index", None),
            attend_to=getattr(task, "attend_to", None),
        )

        print(f"Generated mask: {mask}")
        print(f"Description: {desc}")

        expected_shape = test["cfg"].task.n_target_data + test["cfg"].task.n_target_theta
        assert mask.shape[0] == expected_shape, f"Mask shape mismatch. Expected {expected_shape}, got {mask.shape[0]}"

        if "expected_mask" in test:
            assert torch.all(
                mask == test["expected_mask"]), f"Mask content mismatch. Expected {test['expected_mask']}, got {mask}"
        elif "expected_count" in test:
            assert torch.sum(mask) == test[
                "expected_count"], f"Mask count mismatch. Expected {test['expected_count']}, got {torch.sum(mask)}"

        assert desc == test["expected_desc"], f"Description mismatch. Expected '{test['expected_desc']}', got '{desc}'"

    print("\nAll mask creation tests passed!")


def test_select_targets_by_mask():
    """Test the select_targets_by_mask function."""
    batch_size = 2
    n_target = 5
    feature_dim = 3
    target_results = torch.randn(batch_size, n_target, feature_dim)

    test_masks = [
        {"name": "All targets", "mask": torch.ones(n_target, dtype=torch.bool)},
        {"name": "No targets", "mask": torch.zeros(n_target, dtype=torch.bool)},
        {"name": "First two targets", "mask": torch.tensor([True, True, False, False, False])},
        {"name": "Alternate targets", "mask": torch.tensor([True, False, True, False, True])}
    ]

    for test in test_masks:
        print(f"\nTesting selection with: {test['name']}")
        mask = test["mask"]
        selected = select_targets_by_mask(target_results, mask)

        print(f"Original shape: {target_results.shape}")
        print(f"Selected shape: {selected.shape}")

        expected_count = torch.sum(mask).item()
        assert selected.shape == (batch_size, expected_count, feature_dim), \
            f"Selected shape mismatch. Expected {(batch_size, expected_count, feature_dim)}, got {selected.shape}"

        selected_indices = torch.where(mask)[0]
        for i, idx in enumerate(selected_indices):
            assert torch.all(selected[:, i] == target_results[:, idx]), \
                f"Selected content mismatch at index {i}"

    print("\nAll selection tests passed!")



if __name__ == "__main__":
    print("===== Testing mask creation =====")
    test_create_target_mask()

    print("\n===== Testing target selection =====")
    test_select_targets_by_mask()

