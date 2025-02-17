mkdir output
cd output

wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/fid-5k-256.npz
wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/fid-50k-256.npz

wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/SiT-XL-2-MG-a
wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/SiT-XL-2-MG-b
wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/SiT-XL-2-MG-c
wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/SiT-XL-2-MG-d
wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/SiT-XL-2-MG-e
wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/SiT-XL-2-MG-f

cat SiT-XL-2-MG-* > SiT-XL-2-MG.pt
rm SiT-XL-2-MG-*

wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/SiT-XL-2-REPA-a
wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/SiT-XL-2-REPA-b
wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/SiT-XL-2-REPA-c
wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/SiT-XL-2-REPA-d
wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/SiT-XL-2-REPA-e
wget https://github.com/tzco/diffusion-wo-cfg/releases/download/1.0/SiT-XL-2-REPA-f

cat SiT-XL-2-REPA-* > SiT-XL-2-REPA.pt
rm SiT-XL-2-REPA-*
