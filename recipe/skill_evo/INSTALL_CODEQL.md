# Install CodeQL (Linux)

```bash
cd ~
wget https://github.com/github/codeql-action/releases/download/codeql-bundle-v2.22.1/codeql-bundle-linux64.tar.gz
tar -xzf codeql-bundle-linux64.tar.gz
echo 'export PATH=$HOME/codeql/codeql:$PATH' >> ~/.bashrc
source ~/.bashrc
codeql version
```
