def two_sum(nums, target):
    check_list = {}
    for i in range(len(nums)):
        diff = target - nums[i]
        if nums[i] in check_list:
            return check_list[nums[i]], i
        else:
            check_list[diff] = i


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def addTwoNumbers(l1,l2):
    """
    linked nodes. create a dummy one and then add in new values
    :param l1:
    :param l2:
    :return:
    """
    left = l1
    right = l2
    sumList = ListNode()
    tempList = sumList
    while True:
        if left.next is None and right.next is not None:
            left.next = ListNode()
        elif left.next is not None and right.next is None:
            right.next = ListNode()
        val = left.val + right.val
        if val >= 10:
            val -= 10
            if left.next is None and right.next is None:
                left.next = ListNode()
                right.next = ListNode()
            elif left.next is None:
                left.next = ListNode()
            left.next.val += 1
        tempList.val = val
        if left.next is None and right.next is None:
            break
        tempList.next = ListNode()
        tempList = tempList.next
        left = left.next
        right = right.next
    return sumList


def longest_substring(s):
    """
    find longest substring that doesnt have a repeating character. I am looking placing all the characters seen in the
    dictionary along with the index. if the index is larger than the current substring length, add one to the current
    length else change the current substring to that. finally check the substring against max length to over ride
    :param s:
    :return:
    """
    substring_length = 0
    seen = {}
    current_substring = 0
    for i in range(len(s)):
        if s[i] in seen:
            str_temp = i - seen[s[i]]
            seen[s[i]] = i
            if str_temp > current_substring:
                current_substring += 1
            else:
                current_substring = str_temp
        else:
            seen[s[i]] = i
            current_substring += 1
        if current_substring > substring_length:
            substring_length = current_substring
    return substring_length


def findMedianSortedArray(nums1,nums2):
    """
    combines the two arrays and then finds the median. doesnt do this efficiently
    :param nums1:
    :param nums2:
    :return:
    """
    total_list = []
    if not nums1:
        total_list = nums2
    elif not nums2:
        total_list = nums1
    else:
        i = 0
        j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                total_list.append(nums1[i])
                i += 1
            else:
                total_list.append(nums2[j])
                j += 1
        total_list += nums1[i:] + nums2[j:]
    print(total_list)
    middle = len(total_list)
    if middle % 2 != 0:
        median = total_list[int(middle/2)]
    else:
        median = (total_list[int(middle/2)-1] + total_list[int(middle/2)])/2
    return median


def longestPalindromenaive(s):
    longest_string = ""
    if s == s[::-1]:
        return s
    for i in range(len(s)-1):
        check_string = center_check(s,i,i)
        if len(check_string) > len(longest_string):
            longest_string = check_string
        check_string = center_check(s,i,i+1)
        if len(check_string) > len(longest_string):
            longest_string = check_string
    return longest_string


def center_check(string, left, right):
    while string[left] == string[right]:
        left -= 1
        right += 1
        if right >= len(string) or left < 0:
            break
    return string[left+1:right]


def convert(s, numRows):
    if numRows == 1:
        return s
    step = numRows + (numRows - 2)
    output_string = s[::step]
    for i in range(1, numRows - 1):
        index = i
        loop_counter = 1
        flip = 0
        while index < len(s):
            output_string += s[index]
            if flip % 2 == 0:
                index = step * loop_counter - i
                flip += 1
            else:
                index = step * loop_counter + i
                loop_counter += 1
                flip += 1
    output_string += s[numRows - 1::step]
    return output_string


def reverse(x):
    if x > 2 ** 31 - 1 or x < -2 ** 31:
        return 0
    reversed_int = str(x)
    if reversed_int[0] == "-":
        reversed_int = "-" + reversed_int[-1:0:-1]
    else:
        reversed_int = reversed_int[::-1]
    if int(reversed_int) > 2 ** 31 - 1 or int(reversed_int) < -2 ** 31:
        return 0
    return int(reversed_int)


def myAtoi(string):
    final_int = ""
    sign = 1
    for i in range(len(string)):
        if string[i] == "-":
            if len(final_int) == 0:
                final_int += " "
                sign = -1
            else:
                break
        elif string[i] == "+":
            if len(final_int) == 0:
                final_int += " "
            else:
                break
        elif string[i].isnumeric():
            final_int += string[i]
        elif string[i] == " " and len(final_int) == 0:
            pass
        else:
            break
    try:
        if int(final_int) * sign > 2 ** 31 - 1:
            final_int = 2 ** 31 - 1
        elif int(final_int) * sign < -2 ** 31:
            final_int = 2 ** 31
        return int(final_int) * sign
    except ValueError:
        return 0


def isPalidrome(x):
    """
    revert = 0
    if x < 0 or (x%10==0 and x!=0):
        return False
    elif x < 10:
        return True
    else:
        while x > revert:
            revert = revert*10 + x%10
            x /= 10
            x = int(x)
        check = x==revert or x==int(revert/10)
    return check
    """
    return str(x) == str(x)[::-1]


def isMatch(x):
    pass


def maxArea(height):
    if len(height) == 2:
        return min(height)
    max_area = 0
    left = 0
    right = len(height) -1
    while left < right:
        if min(height[left],height[right]) * (right-left) > max_area:
            max_area = min(height[left],height[right]) * (right-left)
        if height[left] > height[right]:
            right -= 1
        else:
            left += 1
    return max_area


def intToRoman(num):
    roman = {1: "I", 5: "V", 10: "X", 50: "L", 100: "C", 500: "D", 1000: "M"}
    r = [1000, 500, 100, 50, 10, 5, 1]
    roman_string = ""
    for i in range(len(r)):
        div = int(num / r[i])
        if div > 0:
            if div == 4:
                if roman_string:
                    if roman_string[-1] == roman[r[i - 1]]:
                        roman_string = roman_string[:-1] + roman[r[i]] + roman[r[i - 2]]
                    else:
                        roman_string += roman[r[i]] + roman[r[i - 1]]
                else:
                    roman_string += roman[r[i]] + roman[r[i - 1]]
            else:
                roman_string += roman[r[i]] * div
            num -= div * r[i]
    return roman_string


def romanToInt(s):
    num = 0
    rom = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    for i in range(len(s) - 1):
        if rom[s[i]] < rom[s[i + 1]]:
            num -= rom[s[i]]
        else:
            num += rom[s[i]]
    num += rom[s[-1]]
    return num


def longestCommonPrefix(strs):
    if not strs:
        return ""
    if len(strs) == 1:
        return strs[0]
    minLen = float("inf")
    for i in strs:
        minLen = min(minLen, len(i))
    low = 1
    high = minLen
    while low <= high:
        middle = int((low + high) / 2)
        if isCommonPrefix(strs, middle):
            low = middle + 1
        else:
            high = middle - 1
    return strs[0][0:int((low + high) / 2)]


def isCommonPrefix(strs, lens):
    str1 = strs[0][0:lens]
    for i in range(1, len(strs)):
        if strs[i][0:lens] != str1:
            return False
    return True


def threeSum(nums):
    output = []
    target = 0
    if len(nums) < 3:
        return []
    nums.sort()
    print(nums)
    for i in range(len(nums)-2):
        j = i + 1
        k = len(nums) - 1
        print(i)
        if nums[i] > 0:
            break
        if nums[i] == nums[i-1]:
            continue
        while j < k:
            if nums[j] + nums[k] + nums[i] == target:
                output.append([nums[i],nums[j],nums[k]])
                j += 1
                k -= 1
                while j < k and nums[j] == nums[j-1]:
                    j += 1
                while j < k and nums[k] == nums[k+1]:
                    k -= 1
            elif nums[j] + nums[k] + nums[i] < 0:
                j += 1
            else:
                k -= 1
    return output


def threeSumClosest(nums,target):
    output = float("inf")
    if len(nums) == 3:
        return sum(nums)
    nums.sort()
    for i in range(len(nums) - 2):
        j = i + 1
        k = len(nums) - 1
        while j < k:
            s = nums[i] + nums[j] + nums[k]
            if s == target:
                return s
            if abs(target - s) < abs(target - output):
                output = s
            if s < target:
                j += 1
            else:
                k -= 1
    return output


def fourSum(nums,target):
    output = []
    if len(nums) < 4:
        return []
    nums.sort()
    for l in range(len(nums) - 3):
        if l > 0 and nums[l] == nums[l - 1]:
            continue
        for i in range(l + 1, len(nums) - 2):
            j = i + 1
            k = len(nums) - 1
            if i > l + 1 and nums[i] == nums[i - 1]:
                continue
            while j < k:
                if nums[j] + nums[k] + nums[i] + nums[l] == target:
                    output.append([nums[l], nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
                elif nums[j] + nums[k] + nums[i] + nums[l] < target:
                    j += 1
                else:
                    k -= 1
    return output


def removeNthFromEnd(head,n):
    seen = []
    temp = head
    while temp:
        seen.append(temp)
        temp = temp.next
    if len(seen) == 1:
        return None
    if n == len(seen):
        return seen[0].next
    if n == 1:
        seen[-2].next = None
    else:
        seen[-n - 1].next = seen[-n + 1]
        seen[-1].next = None
    return head


def mergeTwoLists(l1,l2):
    head = ListNode()
    dummy = head
    left = l1
    right = l2
    while left and right:
        if left.val < right.val:
            dummy.next = left
            left = left.next
        else:
            dummy.next = right
            right = right.next
        dummy = dummy.next
    if left:
        dummy.next = left
    elif right:
        dummy.next = right
    return head.next


def generateParenthesis(n):
    output = []


    def backTrack(s="", left=0, right=0):
        if len(s) == 2 * n:
            output.append(s)
            return
        if left < n:
            backTrack(s + '(', left + 1, right)
        if right < left:
            backTrack(s + ')', left, right + 1)

    backTrack()
    return output


def mergekLists(lists):
    ans = ListNode()
    output = ans
    vals = []
    i = 0
    while i < len(lists):
        if lists[i]:
            vals.append(lists[i].val)
            i += 1
        else:
            lists.pop(i)
    while vals:
        lowest = min(vals)
        index = vals.index(lowest)
        output.next = ListNode(lowest)
        lists[index] = lists[index].next
        output = output.next
        if not lists[index]:
            lists.pop(index)
        vals = list(a.val for a in lists)
    return ans.next


if __name__ == "__main__":
    """
    a = "babad"
    b = "cbbd"
    c = "abb"
    d = "bba"
    e = "abcba"
    f = "abcda"
    g = "aaabaaaa"
    h = ""
    print(longestPalindromenaive(a)=="bab"or longestPalindromenaive(a)=="aba")
    print(longestPalindromenaive(b)=="bb")
    print(longestPalindromenaive(c)=="bb")
    print(longestPalindromenaive(d)=="bb")
    print(longestPalindromenaive(e)==e)
    print(longestPalindromenaive(f) in f)
    print(longestPalindromenaive(g)=="aaabaaa")
    print(longestPalindromenaive(h)=="")"""
    """
    a = "PAYPALISHIRING"
    numRows = 3
    print(convert(a,numRows)=="PAHNAPLSIIGYIR")
    numRows = 4
    print(convert(a,numRows)=="PINALSIGYAHRPI")"""
