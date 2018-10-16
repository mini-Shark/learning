'''
some concepts : working directory -> stage -> repository
git init : create a new git repository
git add xxx.txt : add a file to stage(暂存区)
git commit -m "some description sentence" : add file from stage to repository
git status : checking the state of current working directory
git diff : check what been modified in current working directory 
git log : display our commit log 
git reset --hard HEAD^ : roll back to previous version, "HEAD" means current version
git reset --hard commit_id : jump to a special version that point by commit_id
git reflog : disply the history of our command, could used to find a commit_id when we close terminal
git diff HEAD -- readme.txt : check the difference between working directory and repository
git checkout -- file.name : discard modify at working directory, actuall it's use repository version to replace working directory version 
git reset HEAD file.name : discard modify at stage and put it back to working directory 
git rm file.name : delete file in repository 
'''