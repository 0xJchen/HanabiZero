#!/bin/bash

echo Input File Prefix
read dirname
find . -type d -name "$dirname" -printf '%Tc %p\n'
