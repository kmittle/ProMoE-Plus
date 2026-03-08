import torch
import torch.nn.functional as F


def _parse_student_with_optional_weight(z_student):
    if torch.is_tensor(z_student):
        return z_student, None
    if isinstance(z_student, (tuple, list)) and len(z_student) == 2:
        z_tilde, token_weight = z_student
        if torch.is_tensor(z_tilde) and torch.is_tensor(token_weight):
            return z_tilde, token_weight
    raise TypeError(
        "Each z_student entry must be either a Tensor or a (Tensor, Tensor) pair "
        "for token-weighted REPA."
    )


def compute_repa_loss(z_teacher, z_student_list):
    """
    Compute REPA projection loss: negative cosine similarity between
    teacher and student representations.

    Args:
        z_teacher: (B, T, D_teacher) teacher features
        z_student_list:
            - list of (B, T, D_proj) projected student features, or
            - list of (z_proj, token_weight) pairs where
              z_proj is (B, T, D_proj) and token_weight is (B, T) or (B, T, 1)
            (one per projector / alignment point)

    Returns:
        proj_loss: scalar mean projection loss
    """
    z_teacher_norm = F.normalize(z_teacher, dim=-1)
    proj_loss = 0.0
    for z_student in z_student_list:
        z_tilde, token_weight = _parse_student_with_optional_weight(z_student)
        z_tilde_norm = F.normalize(z_tilde, dim=-1)
        token_loss = -(z_teacher_norm * z_tilde_norm).sum(dim=-1)  # (B, T)
        if token_weight is not None:
            if token_weight.dim() == 3 and token_weight.shape[-1] == 1:
                token_weight = token_weight.squeeze(-1)
            if token_weight.shape != token_loss.shape:
                raise ValueError(
                    f"token_weight shape {token_weight.shape} must match token loss shape {token_loss.shape}"
                )
            token_loss = token_loss * token_weight
        proj_loss += token_loss.mean()
    proj_loss /= len(z_student_list)
    return proj_loss
