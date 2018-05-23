echo "Build operators for CPU mode"
cd ./mobula_op
make
read -r -p "Whether to build operators for GPU mode? [Y/n] " input
case $input in [yY][eE][sS]|[yY])
    make cuda;;
esac
echo "Finished :-)"
