import torch


def float_greater_than(
    input_tensor: torch.Tensor,
    threshold_tensor: torch.Tensor,
    sigmoid_epsilon: int = 1e7,
) -> torch.Tensor:
    return torch.abs(
        torch.round(torch.sigmoid((input_tensor - threshold_tensor) * sigmoid_epsilon))
    )


def float_lower_than(
    input_tensor: torch.Tensor,
    threshold_tensor: torch.Tensor,
    sigmoid_epsilon: int = 1e7,
) -> torch.Tensor:
    return torch.abs(
        torch.round(torch.sigmoid(-(input_tensor - threshold_tensor) * sigmoid_epsilon))
    )


def float_greater_than_or_equal(
    input_tensor: torch.Tensor,
    threshold_tensor: torch.Tensor,
    sigmoid_epsilon: int = 1e7,
) -> torch.Tensor:
    ot = input_tensor - threshold_tensor
    return torch.abs(
        torch.round(
            torch.sigmoid((ot + torch.abs(torch.min(ot) * 1e-5)) * sigmoid_epsilon)
        )
    )


def float_lower_than_or_equal(
    input_tensor: torch.Tensor,
    threshold_tensor: torch.Tensor,
    sigmoid_epsilon: int = 1e7,
) -> torch.Tensor:
    ot = -(input_tensor - threshold_tensor)
    return torch.abs(
        torch.round(
            torch.sigmoid((ot + torch.abs(torch.min(ot) * 1e-5)) * sigmoid_epsilon)
        )
    )


def float_not(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.round(-(input_tensor - 1)))


def float_and(
    input_tensor_one: torch.Tensor,
    input_tensor_two: torch.Tensor,
    sigmoid_epsilon: int = 1e7,
) -> torch.Tensor:
    return torch.abs(
        torch.round(
            torch.sigmoid(
                ((input_tensor_one + input_tensor_two) - 1.5) * sigmoid_epsilon
            )
        )
    )


def float_or(
    input_tensor_one: torch.Tensor,
    input_tensor_two: torch.Tensor,
    sigmoid_epsilon: int = 1e7,
) -> torch.Tensor:

    return torch.abs(
        torch.round(
            torch.sigmoid(
                ((input_tensor_one + input_tensor_two) - 0.5) * sigmoid_epsilon
            )
        )
    )


def float_equal_0(
    input_tensor: torch.Tensor, sigmoid_epsilon: int = 1e7
) -> torch.Tensor:
    return torch.sigmoid(
        (torch.sigmoid(-torch.abs(input_tensor) * sigmoid_epsilon) * 4 - 1)
        * sigmoid_epsilon
    )
