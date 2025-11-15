#!/bin/bash

USERNAME=$(whoami)

echo "Monitoring jobs for user: $USERNAME"
echo "Press Ctrl+C to stop."

while true; do
    clear
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] Jobs for user: $USERNAME"
    echo "========================================="

    sq | awk -v user="$USERNAME" '
        NR==1 { next }          # on saute l en-tete original
        $2 == user {
            state = $5
            if (state == "R") state_text = "en cours d execution"
            else if (state == "PD") state_text = "en attente"
            else if (state == "CG") state_text = "en cours de completion"
            else if (state == "CD") state_text = "termine"
            else if (state == "CA") state_text = "annule"
            else if (state == "F") state_text = "echoue"
            else state_text = "etat " state
            
            printf "Le job %s (%s) est %s", $1, $4, state_text
            if ($6 != "") printf " - temps restant: %s", $6
            printf "\n"
        }
    '
    
    # Afficher un message si aucun job n'est trouvé
    job_count=$(sq | awk -v user="$USERNAME" 'NR>1 && $2==user {count++} END {print count+0}')
    if [ "$job_count" -eq 0 ]; then
        echo "Aucun job en cours pour cet utilisateur."
    fi

    sleep 5
done
