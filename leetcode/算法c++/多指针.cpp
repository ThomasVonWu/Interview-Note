/* From Momenta AI架构师
题目：
    将负数移动到数组的前面, 0放在中间, 正数移到后面的方法.
例子：
    1 -2 0 -4 5 0 -6 7 -8 0
    =====>
    -2 -4 -8 -6 0 0 0 7 5 1

思路：
    三路划分通常用于快速排序算法中，用来减少比较次数。
    它会把数组分成三个部分：小于某个值的，等于该值的，以及大于该值的。
*/

#include <iostream>
#include <vector>

void moveNegativeZeroPositive(std::vector<int> &nums)
{
    int low = 0, mid = 0, high = nums.size() - 1;
    while (mid <= high)
    {
        if (nums[mid] < 0)
        {
            std::swap(nums[low], nums[mid]);
            ++low;
            ++mid;

            std::cout << "step=1 ";
            for (auto x : nums)
            {
                std::cout << x << " ";
            }
            std::cout << "low=" << low << " mid=" << mid << " high=" << high;
            std::cout << std::endl;
        }
        else if (nums[mid] == 0)
        {
            ++mid;

            std::cout << "step=2 ";
            for (auto x : nums)
            {
                std::cout << x << " ";
            }
            std::cout << "low=" << low << " mid=" << mid << " high=" << high;
            std::cout << std::endl;
        }
        else
        {
            std::swap(nums[mid], nums[high]);
            --high;

            std::cout << "step=3 ";
            for (auto x : nums)
            {
                std::cout << x << " ";
            }
            std::cout << "low=" << low << " mid=" << mid << " high=" << high;
            std::cout << std::endl;
        }
    }
}

int main()
{
    // std::vector<int> nums = {-3, -2, 0, 5, 4}; // 当前方法会改变已经满足要求的数组顺序
    std::vector<int> nums = {1, -2, 0, -4, 5, 0, -6, 7, -8, 0};
    // std::vector<int> nums = {-1, 2, 0, -3, 4, 0, -5, 6};
    std::cout
        << "Ori Array: " << std::endl;
    for (int num : nums)
    {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    moveNegativeZeroPositive(nums);

    std::cout << "Array after moving: " << std::endl;
    for (int num : nums)
    {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}