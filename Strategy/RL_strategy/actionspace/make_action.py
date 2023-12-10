import sys

def create_enum_from_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        file.write("from enum import Enum\n\n")
        file.write("class Actions(Enum):\n")
        for line in lines:
            action = line.strip().replace('-', '_')
            enum_name = ''.join(word.capitalize() for word in action.split('_'))
            file.write(f'    {enum_name} = "-{line.strip()}"\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_action.py <flags.txt> <output_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        create_enum_from_file(input_file, output_file)
