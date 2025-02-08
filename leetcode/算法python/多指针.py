""" From Momenta AI架构师
题目：
    将负数移动到数组的前面, 0放在中间, 正数移到后面的方法.
例子：
    1 -2 0 -4 5 0 -6 7 -8 0
    =====>
    -2 -4 -6 -8 0 0 0 5 7 1
"""


def rearrange_array(nums):
    low, mid, high = 0, 0, len(nums) - 1

    while mid <= high:
        if nums[mid] < 0:
            nums[mid], nums[low] = nums[low], nums[mid]
            low += 1
            mid += 1
        elif nums[mid] == 0:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1

    return nums


if __name__ == "__main__":
    # nums = [1, -2, 0, -4, 5, 0, -6, 7, -8, 0]
    nums = [-3, -2, 0, 5, 4]
    # nums = [-1, 2, 0, -3, 4, 0, -5, 6]

    rearrange_array(nums)
    print(nums)
