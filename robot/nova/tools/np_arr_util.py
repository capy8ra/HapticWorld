def pretty_arr(arr):
    return "".join([f"  ({i+1}){arr[i]:8.4f}  " for i in range(len(arr))])