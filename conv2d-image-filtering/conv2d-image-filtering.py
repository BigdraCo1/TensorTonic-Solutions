def conv2d(image: list, kernel: list, stride: int = 1, padding: int = 0) -> list:
    """
    Returns a two-dimensional list.
    """
    x_in = len(image[0])
    y_in = len(image)
    kx = len(kernel[0])
    ky = len(kernel)
    x_out = (x_in + 2*padding - kx)//stride + 1
    y_out = (y_in + 2*padding - ky)//stride + 1
    padded_x = x_in + 2*padding
    padded_y = y_in + 2*padding
    side = [0] * padding
    border = [[0] * padded_x for _ in range(padding)]
    padded_image = border + [side + row + side for row in image] + border
    output = []
    for y in range(y_out):
        row_output = []
        for x in range(x_out):
            sum_point = 0
            for j in range(ky):
                sum_row = 0
                for i in range(kx):
                    sum_row += kernel[j][i] * padded_image[y*stride + j][x*stride + i]
                sum_point += sum_row
            row_output.append(sum_point)
        output.append(row_output)

    return output