#!/bin/bash
# Bash Menu Script Example
echo "Edit params.ipynb to choose your custom parameters"
PS3='Please enter your choice: '
options=("2D2V(Ex, Ey, Bz)" "2D3V(Ex, Ey, Ez, Bx, By, Bz)" )
select opt in "${options[@]}"
do
    case $opt in
        "2D2V(Ex, Ey, Bz)")
            echo "you chose 2D2V"
            /home/tejas/anaconda3/bin/jupyter-notebook /home/tejas/Documents/2DPIC_edit2/2D2V_2D3V/2D2V.ipynb
            ;;
        "2D3V(Ex, Ey, Ez, Bx, By, Bz)")
            echo "you chose 2D3V"
            /home/tejas/anaconda3/bin/jupyter-notebook /home/tejas/Documents/2DPIC_edit2/2D2V_2D3V/2D3V.ipynb 
            ;;
        *) echo invalid option;;
    esac
done