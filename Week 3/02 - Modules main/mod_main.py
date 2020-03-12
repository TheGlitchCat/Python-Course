# Main Example
from time import sleep


def wait():
    sleep(4)


if __name__ == '__main__':
    print('Run Directly')
    wait()
    print('Initial Config done')
else:
    print('Run from import')
    wait()
    print('Import Config done')


print(f'{__name__} Loaded')

'''
def main():
    print('Run Directly')
    
    
if __name__ == '__main__':
    main()
'''
