#!/bin/bash
cd ../../../
echo "directory: $(pwd)"
tar_file="_site.tar.gz"
if [ -e $tar_file ]; then
    echo "Found: _site.tar.gz"
    rm -rf $tar_file
    echo "Remove _site.tar.gz"
fi
echo "Create tar file..."
tar -zcvf $tar_file _site
echo "Deploy to server..."
scp $tar_file xt@xta0.me:~/

if [ "$?" -eq "0" ]; then
    echo "Upload Succeed"
else
    echo "Upload Failed!"
fi
echo "Clean up files..."
rm -rf $tar_file
echo "Done."
