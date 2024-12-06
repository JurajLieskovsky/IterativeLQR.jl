import re
import csv
import argparse

# Create a command-line argument parser
parser = argparse.ArgumentParser(description="Extract numbers from an input file and save them to a CSV file.")
parser.add_argument("input_file", help="Path to the input file")
parser.add_argument("output_file", help="Path to the output file")
parser.add_argument(
    "--skip",
    type=int,
    default=0,
    help="Number of lines to skip from the start of the file",
)

# Parse the command-line arguments
args = parser.parse_args()

# Extract input and output file paths from the parsed arguments
input_file = args.input_file
output_file = args.output_file
lines_to_skip = args.skip

with open(input_file, "r") as input, open(output_file, "w", newline="") as output:
    writer = csv.writer(output)

    # Skip the specified number of lines
    for _ in range(lines_to_skip):
        next(input)

    # Read the first line after skipping
    first_line = input.readline().strip()
    column_names = re.findall(r"(\w+)\s*=\s*[-+]?(?:\d+(?:\.\d+)?)", first_line)
    writer.writerow(column_names)

    # read data from remaining lines
    for line in input:
        # Extract numbers from each line
        number_strings = re.findall(r"([-+]?\d+(?:\.\d+)?)(?=\s|,|$)", line)

        # Convert the extracted strings to floating-point numbers
        numbers = [float(num) if "." in num else int(num) for num in number_strings]

        # Write the extracted numbers to the output file as a row
        writer.writerow(numbers)

print(f"Numbers have been extracted from '{input_file}' and saved to '{output_file}'.")
