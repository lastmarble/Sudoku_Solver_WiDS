from solve_puzzle import board


# to print the Grid

def print_grid(arr):
    for i in range(9):
        for j in range(9):
            print(arr[i][j], end=" ")
        print()


def find_empty(arr, location):
    for row in range(9):
        for col in range(9):
            if arr[row][col] == 0:
                location[0] = row
                location[1] = col
                return True
    return False


def used_in_row_col_box(arr, row, col, num):
    for i in range(9):
        if arr[row][i] == num:
            return True
        if arr[i][col] == num:
            return True
    for k in range(3):
        for j in range(3):
            if arr[k + row - row % 3][j + col - col % 3] == num:
                return True
    return False


def is_safe(arr, row, col, num):
    return not used_in_row_col_box(arr, row, col, num)


def solve_sudoku(arr):
    location = [0, 0]

    if not find_empty(arr, location):
        return True
    row = location[0]
    col = location[1]

    # digits 1 to 9
    for num in range(1, 10):
        # if safe
        if is_safe(arr, row, col, num):
            arr[row][col] = num

            # return, if success
            if solve_sudoku(arr):
                return True

            # false - make zero again
            arr[row][col] = 0

    return False


if solve_sudoku(board):
    print("Solved :")
    print_grid(board)
else:
    print("No solution exists")
