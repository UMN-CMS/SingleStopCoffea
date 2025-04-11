#python3 -m analyzer run-samples -s DataSingleMuon2018Official -o "DataSingleMuon2018Official.pkl" -m objects baseline_muon_selection apply_selection trigger_efficiency_hists jets -a localhost:10006

#python3 -m analyzer run-samples -s DataSingleMuon2017 -o "DataSingleMuon2017.pkl" -m objects baseline_muon_selection apply_selection trigger_efficiency_hists jets -a localhost:10006

#python3 -m analyzer run-samples -s DataSingleMuon2016_preVFP -o "DataSingleMuon2016_preVFP.pkl" -m objects baseline_muon_selection apply_selection trigger_efficiency_hists jets -a localhost:10006

python3 -m analyzer run-samples -s DataSingleMuon2016_postVFP -o "DataSingleMuon2016_postVFP.pkl" -m objects baseline_muon_selection apply_selection trigger_efficiency_hists jets -a localhost:10006

#python3 -m analyzer run-samples -s DataMuon2023 -o "DataMuon2023.pkl" -m objects baseline_muon_selection apply_selection trigger_efficiency_hists jets -a localhost:10006

#python3 -m analyzer run-samples -s DataMuon2022PreEE -o "DataMuon2022PreEE.pkl" -m objects baseline_muon_selection apply_selection trigger_efficiency_hists jets -a localhost:10006

#python3 -m analyzer run-samples -s DataMuon2022PostEE -o "DataMuon2022PostEE.pkl" -m objects baseline_muon_selection apply_selection trigger_efficiency_hists jets -a localhost:10006
