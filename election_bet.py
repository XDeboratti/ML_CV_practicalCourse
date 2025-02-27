#Union, Afd, SPD, Grüne, Linke, BSW, FDP
results = [28.52, 20.8, 16.41, 11.61, 8.77, 4.97, 4.33]
maike = [25.0, 19.0, 13.0, 10.0, 8.0, 6.0, 5.0]
manu = [30.5, 21.5, 15.0, 12.5, 7.5, 5.0, 4.5]
klaus = [31.0, 25.0, 15.0, 14.0, 6.0, 4.5, 5.5]
sven = [30.0, 20.0, 16.0, 13.0, 6.0, 4.0, 5.0]
debo = [31.0, 22.0, 14.0, 13.0, 5.0, 4.0, 6.0]

bets = [maike, manu, klaus, sven, debo]
diffs = []

for bet in bets:
    diff = 0.0
    for i in range(0, len(results)):
        diff += abs(bet[i]  - results[i])
    print(diff)
    diffs.append(diff)

bets_asc_diff = [val for _, val in sorted(zip(diffs, bets))]
print(bets_asc_diff)

#evaluate per party, evaluate l2 norm, plot results and our tips, plot diffs per tip, plot siegertreppchen, 