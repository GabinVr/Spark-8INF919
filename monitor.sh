#!/bin/bash

USERNAME=$(whoami)

echo "Monitoring jobs for user: $USERNAME"
echo "Press Ctrl+C to stop."

while true; do
    clear
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] Jobs for user: $USERNAME"
    echo "========================================="

    # On parcourt les jobs de l'utilisateur avec awk et on affiche aussi la fin du fichier slurm-$JOBID.out
    sq | awk -v user="$USERNAME" 'NR==1 { next } $2 == user { print $1, $4, $5, $6 }' | while read -r JOB_ID JOB_NAME JOB_STATE TIME_LEFT; do
        # Traduction de l'état
        case "$JOB_STATE" in
            R)  STATE_TEXT="en cours d execution" ;;
            PD) STATE_TEXT="en attente" ;;
            CG) STATE_TEXT="en cours de completion" ;;
            CD) STATE_TEXT="termine" ;;
            CA) STATE_TEXT="annule" ;;
            F)  STATE_TEXT="echoue" ;;
            *)  STATE_TEXT="etat $JOB_STATE" ;;
        esac

        # Ligne d'état du job
        if [ -n "$TIME_LEFT" ]; then
            echo "Le job $JOB_ID ($JOB_NAME) est $STATE_TEXT - temps restant: $TIME_LEFT"
        else
            echo "Le job $JOB_ID ($JOB_NAME) est $STATE_TEXT"
        fi

        # Affichage des dernières lignes du fichier de log associé
        LOG_FILE="slurm-${JOB_ID}.out"
        if [ -f "$LOG_FILE" ]; then
            echo "--- Dernières lignes de $LOG_FILE ---"
            tail -n 5 "$LOG_FILE"
        else
            echo "(Fichier de log $LOG_FILE non encore créé)"
        fi
        echo "-----------------------------------------"
    done

    # Afficher un message si aucun job n'est trouvé
    job_count=$(sq | awk -v user="$USERNAME" 'NR>1 && $2==user {count++} END {print count+0}')
    if [ "$job_count" -eq 0 ]; then
        echo "Aucun job en cours pour cet utilisateur."
    fi

    sleep 5
done
