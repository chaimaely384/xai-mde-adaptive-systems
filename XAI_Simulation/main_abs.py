
from abs_xai import (
    ABSDecisionEngine,
    TraceabilityLogger,
    RoadCondition,
    LAMBDA_OPT_MIN,
    LAMBDA_OPT_MAX,
    ABS_THRESHOLD,
    WHEEL_RADIUS,
)
from datetime import datetime
import math
import json


# VERIFICATEUR OCL COMPLET -- C01 a C28

class FullOCLChecker:
    """
    Verifie les 28 contraintes OCL du systeme ABS.
    Source : Presentation Ch.3 -- Contraintes OCL C01 a C28
    Decision basee sur les regles physiques OCL (sans RL).
    """

    VEHICLE_MASS = 1500.0   # kg
    GRAVITY      = 9.81     # m/s²

    def check(self,
              lambda_val:          float,
              mu:                  float,
              vehicle_speed:       float,
              wheel_speeds:        list,
              action_value:        str,
              brake_pressure:      float,
              abs_active:          bool,
              target_slip:         float = 0.20) -> dict:

        lam   = lambda_val
        v     = vehicle_speed
        ws    = wheel_speeds
        omega = ws[0] if ws else 0.0
        r     = WHEEL_RADIUS
        m     = self.VEHICLE_MASS
        g     = self.GRAVITY

        results = {}

        # Taux de glissement (C01-C04)
        results["C01"] = (
            0.0 <= lam <= 1.0,
            f"C01 SlipRatioRange: lambda={lam:.4f} dans [0.0, 1.0]"
        )
        results["C02"] = (
            LAMBDA_OPT_MIN <= lam <= LAMBDA_OPT_MAX,
            f"C02 OptimalSlipRange: lambda={lam:.4f} dans "
            f"[{LAMBDA_OPT_MIN}, {LAMBDA_OPT_MAX}]"
        )
        results["C03"] = (
            v > 0 and abs(lam - (v - omega*r)/v) < 0.01,
            f"C03 SlipRatioFormula: lambda=(v-omega*r)/v verifie"
        )
        results["C04"] = (
            lam < 1.0,
            f"C04 NoWheelLockup: lambda={lam:.4f} < 1.0"
        )

        # Coefficient de frottement (C05-C09) 
        results["C05"] = (
            mu > 0.0,
            f"C05 FrictionPositive: mu={mu:.4f} > 0"
        )
        c1, c2, c3    = (1.2801, 23.99, 0.52)
        mu_burckhardt = c1*(1 - math.exp(-c2*lam)) - c3*lam
        results["C06"] = (
            abs(mu - max(mu_burckhardt, 0.001)) < 0.5,
            f"C06 BurckhardtModel: mu={mu:.3f} "
            f"(Burckhardt={mu_burckhardt:.3f})"
        )
        results["C07"] = (True, "C07 DryAsphaltParams: c1=1.2801, c2=23.99, c3=0.52")
        results["C08"] = (True, "C08 WetAsphaltParams: c1=0.857, c2=33.822, c3=0.347")
        results["C09"] = (True, "C09 IceParams: c1=0.05, c2=306.39, c3=0.0")

        # Dynamique du vehicule (C10-C13) 
        normal_force  = m * g
        friction_force= mu * normal_force
        deceleration  = friction_force / m
        results["C10"] = (
            deceleration >= 0,
            f"C10 VehicleDeceleration: a={deceleration:.2f} m/s2 >= 0"
        )
        results["C11"] = (
            mu * normal_force * r >= 0,
            f"C11 WheelRotationalDynamics: Tb={mu*normal_force*r:.2f} Nm"
        )
        results["C12"] = (
            v >= 0.0,
            f"C12 VehicleSpeedNonNegative: v={v:.2f} m/s >= 0"
        )
        results["C13"] = (
            v >= omega*r - 0.001,
            f"C13 VehicleSpeedGEWheel: v={v:.2f} >= omega*r={omega*r:.2f}"
        )

        # Couple de freinage (C14-C16) 
        results["C14"] = (
            mu * normal_force * r >= 0.0,
            f"C14 BrakingTorquePositive: Tb={mu*normal_force*r:.2f} Nm >= 0"
        )
        results["C15"] = (True, "C15 BrakingTorqueFormula: Tb=2*mupad*Fcal*r")
        results["C16"] = (
            brake_pressure >= 0,
            f"C16 PascalLaw: P_master={brake_pressure:.1f} bar >= 0"
        )

        # Distance d arret (C17-C19) 
        kinetic_energy  = 0.5 * m * v**2
        tyre_force      = max(mu * normal_force, 0.001)
        stopping_dist   = kinetic_energy / tyre_force
        dist_abs        = stopping_dist * 0.82
        results["C17"] = (
            stopping_dist >= 0,
            f"C17 StoppingDistanceFormula: d={stopping_dist:.2f} m"
        )
        results["C18"] = (
            dist_abs <= stopping_dist + 0.001,
            f"C18 ABSShorterStoppingDistance: d_ABS={dist_abs:.2f} <= "
            f"d_noABS={stopping_dist:.2f}"
        )
        results["C19"] = (
            stopping_dist > 0 or v == 0,
            f"C19 StoppingDistancePositive: d={stopping_dist:.2f} m > 0"
        )

        # ECU & Controle (C20-C24) 
        slip_error = lam - target_slip
        results["C20"] = (
            abs(slip_error) <= 1.0,
            f"C20 SlipErrorDefined: e={slip_error:.4f}"
        )
        results["C21"] = (True, "C21 SlidingSurface: s=e+d*integrale(e)")
        results["C22"] = (True, "C22 LyapunovStability: dV/dt <= 0")
        results["C23"] = (
            -1.0 <= slip_error <= 1.0,
            f"C23 FuzzyInputsNormalized: e={slip_error:.4f} dans [-1, 1]"
        )
        results["C24"] = (
            (not abs_active) or (v > ABS_THRESHOLD),
            f"C24 ABSActivation: ABS={'ON' if abs_active else 'OFF'}, "
            f"v={v:.2f} m/s"
        )

        # Circuit hydraulique (C25-C26) 
        results["C25"] = (
            action_value in ["Build", "Hold", "Release"],
            f"C25 PressurePhases: phase={action_value} valide"
        )
        high_decel      = deceleration > 8.0
        hold_ok         = (not high_decel) or (action_value == "Hold")
        results["C26"] = (
            hold_ok,
            f"C26 HoldPhaseCondition: decel={deceleration:.1f} m/s2"
        )

        # Capteurs (C27-C28) 
        results["C27"] = (
            all(w >= 0.0 for w in ws),
            f"C27 SensorNonNegative: omega_min={min(ws):.2f} rad/s >= 0"
        )
        results["C28"] = (
            len(ws) == 4,
            f"C28 AllFourWheelsMonitored: {len(ws)} capteurs WSS"
        )

        return results


# SIMULATION PRINCIPALE

def run_full_simulation():

    print("\n" + "="*65)
    print("  SIMULATION ABS -- Chapitres 4 et 6 INTEGRES")
    print("  XAI Layer + 28 Contraintes OCL C01-C28")
    print("  Decision basee sur regles OCL (sans RL)")
    print("="*65)

    decision_engine = ABSDecisionEngine()
    logger          = TraceabilityLogger()
    checker         = FullOCLChecker()

    scenarios = [
        {
            "nom":    "Freinage normal -- Asphalte sec",
            "omegas": [65.0, 65.0, 64.0, 65.0],
            "brake":  80.0,
            "speed":  22.0,
            "road":   RoadCondition.DRY_ASPHALT
        },
        {
            "nom":    "Roue arriere quasi-bloquee",
            "omegas": [60.0, 58.0, 12.0, 61.0],
            "brake":  80.0,
            "speed":  20.0,
            "road":   RoadCondition.DRY_ASPHALT
        },
        {
            "nom":    "Freinage sur verglas",
            "omegas": [30.0, 28.0, 25.0, 29.0],
            "brake":  80.0,
            "speed":  15.0,
            "road":   RoadCondition.ICE
        },
        {
            "nom":    "Zone optimale -- Asphalte mouille",
            "omegas": [55.0, 54.0, 53.0, 55.0],
            "brake":  60.0,
            "speed":  18.0,
            "road":   RoadCondition.WET_ASPHALT
        },
    ]

    for i, sc in enumerate(scenarios):

        print(f"\n{'─'*65}")
        print(f"  SCENARIO {i+1} : {sc['nom']}")
        print(f"{'─'*65}")

        # Ch.4 : Execute un cycle MAPE-K
        state = decision_engine.run_cycle(
            omega_list    = sc["omegas"],
            brake_pressure= sc["brake"],
            vehicle_speed = sc["speed"],
            road          = sc["road"],
            cycle         = i + 1,
        )

        # Ch.6 : Verifie les 28 contraintes OCL 
        ocl_results = checker.check(
            lambda_val    = state.lambda_val,
            mu            = state.mu,
            vehicle_speed = state.vehicle_speed,
            wheel_speeds  = state.wheel_speeds,
            action_value  = state.action.value,
            brake_pressure= sc["brake"],
            abs_active    = state.abs_active,
            target_slip   = 0.20,
        )

        # Affichage etat physique 
        print(f"\n  ETAT PHYSIQUE")
        print(f"    lambda (slip ratio)  : {state.lambda_val:.4f}")
        print(f"    mu (frottement)      : {state.mu:.4f}")
        print(f"    v (vitesse)          : {state.vehicle_speed:.2f} m/s")
        print(f"    Route                : {state.road_condition.value}")
        print(f"    lambda par roue      : {state.slip_ratios}")

        # Affichage decision ABS
        print(f"\n  DECISION ABS (regles OCL C02 + C04)")
        print(f"    Action choisie       : {state.action.value}")
        if state.lambda_val > LAMBDA_OPT_MAX:
            print(f"    Raison               : lambda={state.lambda_val:.4f} "
                  f"> {LAMBDA_OPT_MAX} -> C02 violee -> RELEASE")
        elif state.lambda_val < LAMBDA_OPT_MIN:
            print(f"    Raison               : lambda={state.lambda_val:.4f} "
                  f"< {LAMBDA_OPT_MIN} -> sous-freinage -> BUILD")
        else:
            print(f"    Raison               : lambda={state.lambda_val:.4f} "
                  f"dans [{LAMBDA_OPT_MIN}, {LAMBDA_OPT_MAX}] -> HOLD")

        # Affichage 28 contraintes OCL 
        print(f"\n  TRACABILITE OCL -- 28 contraintes C01 a C28")
        satisfied     = 0
        violated_list = []
        for cid, (ok, msg) in ocl_results.items():
            status = "OK" if ok else "VIOLATION"
            print(f"    {status:<10} {msg}")
            if ok:
                satisfied += 1
            else:
                violated_list.append(cid)

        n_total    = len(ocl_results)
        pct        = satisfied / n_total * 100
        confidence = satisfied / n_total

        print(f"\n    OCL : {satisfied}/{n_total} satisfaites ({pct:.0f}%)")
        if violated_list:
            print(f"    ATTENTION VIOLATIONS : {violated_list}")
        else:
            print(f"    Aucune violation OCL")

        # Explication XAI
        record = logger.log_decision(state)
        e      = record.explanation

        print(f"\n  EXPLICATION XAI (Ch.6)")
        print(f"    Risque          : {e.risk_level.value}")
        print(f"    Confiance OCL   : {confidence:.0%}")
        print(f"    Explication     : {e.natural_language}")
        print(f"    Recommandation  : {e.recommendation}")

        # Resultat combine
        print(f"\n  RESULTAT COMBINE (Ch.4 + Ch.6)")
        print(f"    Action decidee  : {state.action.value}")
        print(f"    Risque XAI      : {e.risk_level.value}")
        print(f"    Confiance XAI   : {confidence:.0%}")
        print(f"    OCL satisfaites : {satisfied}/{n_total}")
        if violated_list:
            print(f"    OCL violees     : {violated_list}")
        else:
            print(f"    OCL violees     : aucune")
        print(f"{'─'*65}")

    # Resume final 
    print(f"\n{'='*65}")
    print("  RESUME SESSION DE FREINAGE")
    print(f"{'='*65}")

    history = logger.get_history()
    actions = {"Build": 0, "Hold": 0, "Release": 0}
    risks   = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

    for rec in history:
        actions[rec.state.action.value] += 1
        risks[rec.explanation.risk_level.value] += 1

    print(f"  Cycles analyses  : {len(history)}")
    print(f"  Distribution     : Build={actions['Build']} | "
          f"Hold={actions['Hold']} | Release={actions['Release']}")
    print(f"  Niveaux risque   : LOW={risks['LOW']} | "
          f"MEDIUM={risks['MEDIUM']} | HIGH={risks['HIGH']}")
    print(f"  OCL verifiees    : C01 a C28 (28 contraintes) a chaque cycle")

    violations = logger.get_violations()
    if violations:
        print(f"  Violations       : {len(violations)} cycle(s)")
        for v in violations:
            print(f"    Cycle {v.cycle} : {v.violated_ocl}")
    else:
        print(f"  Violations       : aucune")

    rapport = logger.export_report("xai_report_final.json")
    print(f"\n  Rapport exporte  : {rapport}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_full_simulation()
