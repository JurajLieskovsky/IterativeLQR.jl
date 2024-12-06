# Stripper (CLI tool)

a simple CLI tool for stripping all non integers or decimal numbers from an input file and writing what remains to a .csv file

## Basis single file usage
```
python stripper.py input.log output.csv --skip 5
```

## By folder (FISH shell)
```fish
for input_file in (find <input_folder> -name "*.<extension>")
   set filename (basename "$input_file" | string replace -r "\.\w+" "")
   python <path/to/stripper>/stripper.py "$input_file" "<output_folder>/$filename.csv" --skip <lines_in_header>
end
```

## By folder (BASH shell)
```bash
for input_file in $(find <input_folder> -name "*.<extension>"); do
   filename=$(basename "$input_file" | sed -E 's/\.[^.]+$//')
   python <path/to/stripper>/stripper.py "$input_file" "<output_folder>/$filename.csv" --skip <lines_in_header>
done
```
