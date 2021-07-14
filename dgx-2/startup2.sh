# HTTP Proxy (on VM) (Step 1)

sudo apt install privoxy
echo forward-socks5  /  127.0.0.1:3128  . | sudo tee -a /etc/privoxy/config
sed -i '$ a alias proxy=$(echo http{,s,2}_proxy=http://127.0.0.1:8118 HTTP{,S,2}_PROXY=http://127.0.0.1:8118)' ~/.bashrc
sed -i '$ a export http{,s,2}_proxy=http://127.0.0.1:8118 HTTP{,S,2}_PROXY=http://127.0.0.1:8118' ~/.bashrc
sudo systemctl restart privoxy
