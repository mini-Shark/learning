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
how to push our local repository to remote github server ?
1. generate a SSH key 
2. add this key at my github account 
3. create a new repo in my github account 
4. connect local repo and remote repo as "git remote add origin git@github.com:mini-Shark/learning.git"
   In this command, "origin" is remote repo name 
5. push all content to remote repo, like this "git push -u origin master"

notes : we both have master branch at local and remote 
'''