import torch
import torch.nn.functional as F


def compute_repa_loss(z_teacher, z_student_list):
    """
    Compute REPA projection loss: negative cosine similarity between
    teacher and student representations.

    Args:
        z_teacher: (B, T, D_teacher) teacher features
        z_student_list: list of (B, T, D_proj) projected student features
                        (one per projector / alignment point)

    Returns:
        proj_loss: scalar mean projection loss
    """
    z_teacher_norm = F.normalize(z_teacher, dim=-1)
    proj_loss = 0.0
    for z_tilde in z_student_list:
        z_tilde_norm = F.normalize(z_tilde, dim=-1)
        proj_loss += -(z_teacher_norm * z_tilde_norm).sum(dim=-1).mean()
    proj_loss /= len(z_student_list)
    return proj_loss
