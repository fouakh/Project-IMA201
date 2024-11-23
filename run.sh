#!/bin/bash

if [ -z "$2" ]; then
  echo "Usage: $0 [on|off] <directory>"
  exit 1
fi

MODE=$1
DIRECTORY=$2

if [ "$DIRECTORY" = "Images" ]; then
  echo "Error: Directory name cannot be 'Images'. Please provide a different directory."
  exit 1
fi

if [ "$MODE" = "on" ]; then
  python -B Undermarine_main.py -s "on" -d "$DIRECTORY"
elif [ "$MODE" = "off" ]; then
  python -B Undermarine_main.py -s "off" -d "$DIRECTORY"
else
  echo "Invalid mode. Use 'on' to process images or 'off' to delete them."
  exit 1
fi
