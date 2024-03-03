# This script is used to reinit the data git and push the representations
excpected_git_remote="git@github.com:JonasLoos/h-space-similarity-explorer-data.git"

# try to open the representations directory
cd representations || exit 1

# check if this is a git repository
if [[ -d .git ]]; then
  # make sure the git remote is correct
  git_remote=$(git remote get-url origin)
  if [[ $git_remote != $excpected_git_remote ]]; then
    echo "Git remote is not 'h-space-similarity-explorer-data'. Exiting."
    exit 1
  fi

  # remove the .git folder
  trash .git
fi

# print size of folder
echo "Size of representations folder: $(du -sh | cut -f1)"

# ask if the user wants to continue
read -p "Do you want to continue? (y/n) " -n 1 -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Exiting."
  exit 1
fi
echo "Pushing representations to git remote $excpected_git_remote"

# reinit git
git init
git remote add origin $excpected_git_remote

# go through all files and folder in the folder and add them to the git
# this is necessary to stay below the 2GB upload limit of Github
for file in *; do
  echo "adding and pushing $file to git with size $(du -sh $file | cut -f1)"
  git add -f $file
  git commit -m "Added $file"
  git push -f -u origin main
done
