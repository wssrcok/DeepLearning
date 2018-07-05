def try_to_change_list_contents(the_list):
    print('got', the_list)
    the_list.update({4:'four'})
    print('changed to', the_list)

outer_list = {1:'one', 2:'two', 3:'three'}

print('before, outer_list =', outer_list)
try_to_change_list_contents(outer_list)
print('after, outer_list =', outer_list)