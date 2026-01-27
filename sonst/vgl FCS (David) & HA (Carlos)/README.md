
Hi Oliver,

Entschuldige die Verspätung. Ich hatte leider viel zu tun, und war auch etwas eigensinnig. Da wir für das nächste Projekt ebenfalls I-V sowie ladungsaufgelösten Kurven für SS junctions benötigen, habe ich gedacht, dass ich einfach einen Code schreibe, der für beides funktioniert.

In dem folgenden Dropbox link kannst du die Daten sehen:
https://cloud.uni-konstanz.de/index.php/f/136669082

Diese sind wie folgt strukturiert:

P1_results_new_<Transmission>_n_20_T_<k_B*Temperatur in units of Delta>_NE_5000_Nchi_50.mat

Die Sachen wie n = 20, N_E = 5000 und N_chi = 50 sind nicht relevant für dich.

In jedem der Dateien gibt es die folgenden Grössen:

eV_eff_range = Die Spannung als Funktion von der Energielücke Delta

I_In = Der Strom in units of eDelta/h. 

S_In = Shotnoise (nicht relevant für dich)

In_final = Ladungsaugelöste Ströme von n = -20 bis n= 20. Also Quasiteiclhen sind bei n = 22. 

Falls du irgendwelche Fragen hast, frage bitte. :)

Beste Grüsse,
David
