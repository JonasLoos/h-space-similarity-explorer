# This script is used to reinit the data git and push the representations

# try to open the representations directory
cd representations || exit 1

# make sure the git remote is correct
git_remote=$(git remote get-url origin)
if [[ $git_remote != *"h-space-similarity-explorer-data"* ]]; then
  echo "Git remote is not 'h-space-similarity-explorer-data'. Exiting."
  exit 1
fi

echo "Pushing representations to git remote $git_remote"

# reinit git
trash .git
git init
git remote add origin $git_remote

# add files and push
git add -f *
git commit -m "Addding representations"
git push -f origin main
