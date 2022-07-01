## Code to Remove Duplicates

string = input()
result = string[0]
def check(string, result):
    string = string[1:]
    try:
        if(result[-1] != string[0]):
            result = result + string[0]
            return check(string, result)
        else:
            return check(string, result)
    except IndexError:
        print(result)
check(string, result)