echo "Muon 2023"
#dasgoclient -query='file dataset=/Muon0/Run2023C-22Sep2023_v1-v1/NANOAOD | sum(file.nevents)' 
#dasgoclient -query='file dataset=/Muon0/Run2023C-22Sep2023_v2-v1/NANOAOD | sum(file.nevents)' 
#dasgoclient -query='file dataset=/Muon0/Run2023C-22Sep2023_v3-v1/NANOAOD | sum(file.nevents)' 
#dasgoclient -query='file dataset=/Muon0/Run2023C-22Sep2023_v4-v1/NANOAOD | sum(file.nevents)' 
#dasgoclient -query='file dataset=/Muon1/Run2023C-22Sep2023_v1-v1/NANOAOD | sum(file.nevents)' 
#dasgoclient -query='file dataset=/Muon1/Run2023C-22Sep2023_v2-v1/NANOAOD | sum(file.nevents)' 
#dasgoclient -query='file dataset=/Muon1/Run2023C-22Sep2023_v3-v1/NANOAOD | sum(file.nevents)' 
#dasgoclient -query='file dataset=/Muon0/Run2023D-22Sep2023_v1-v1/NANOAOD | sum(file.nevents)' 
#dasgoclient -query='file dataset=/Muon0/Run2023D-22Sep2023_v2-v1/NANOAOD | sum(file.nevents)' 
#dasgoclient -query='file dataset=/Muon1/Run2023D-22Sep2023_v1-v1/NANOAOD | sum(file.nevents)' 
#dasgoclient -query='file dataset=/Muon1/Run2023D-22Sep2023_v2-v1/NANOAOD | sum(file.nevents)' 

echo "Muon 2022 Post EE"
#dasgoclient -query='file dataset=/Muon/Run2022E-22Sep2023-v1/NANOAOD | sum(file.nevents)' 
#dasgoclient -query='file dataset=/Muon/Run2022F-22Sep2023-v2/NANOAOD | sum(file.nevents)' 
#dasgoclient -query='file dataset=/Muon/Run2022G-22Sep2023-v1/NANOAOD | sum(file.nevents)' 

echo "Muon 2022 Pre EE"
dasgoclient -query='file dataset=/Muon/Run2022C-22Sep2023-v1/NANOAOD | sum(file.nevents)' 
dasgoclient -query='file dataset=/Muon/Run2022D-22Sep2023-v1/NANOAOD | sum(file.nevents)' 
