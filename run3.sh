signals=(
    signal_312_1000_400
    signal_312_1000_600
    signal_312_1000_900
    signal_312_1200_400
    signal_312_1200_600
    signal_312_1200_1100
    signal_312_1400_400
    signal_312_1400_600
    signal_312_1400_1300
    signal_312_1500_400
    signal_312_1500_600
    signal_312_1500_900
    signal_312_1500_1400
    signal_312_2000_400
    signal_312_2000_600
    signal_312_2000_900
    signal_312_2000_1400
    signal_312_2000_1900
    signal_312_1000_700
    signal_312_1000_800
    signal_312_1200_700
    signal_312_1200_800
    signal_312_1200_900
    signal_312_1200_1000
    signal_312_1500_1000
    signal_312_1500_1100
    signal_312_1500_1200
    signal_312_1500_1300
    signal_312_1500_1350
    signal_312_1500_1450
    signal_312_2000_1200
    signal_312_2000_1300
    signal_312_2000_1500
    signal_312_2000_1600
    signal_312_2000_1700
    signal_312_200_100
    signal_312_300_100
    signal_312_300_200
    signal_312_500_100
    signal_312_500_200
    signal_312_500_400
    signal_312_700_100
    signal_312_700_400
    signal_312_700_600
    signal_312_800_400
    signal_312_800_600
    signal_312_800_700
    signal_312_900_400
    signal_312_900_600
    signal_312_900_700
    signal_312_900_800
)

for i in "${signals[@]}"; do
	echo $i
	#python3 -m analyzer run-samples -s $i -o "Run3/$i.pkl" -m objects run3_hists
done

#python3 -m analyzer run-samples -s QCDInclusive2023 -o "Run3/QCDInclusive2023.pkl" -m objects run3_hists -a localhost:10006

python3 -m analyzer run-samples -s QCDInclusive2023 signal_312_200_100 signal_312_300_200 signal_312_500_100 signal_312_700_400 signal_312_900_600 signal_312_1000_900 signal_312_1500_600 signal_312_1500_900 signal_312_2000_1900 -o "Run3/combined.pkl" -m objects run3_hists -a localhost:10006
