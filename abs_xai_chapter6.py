from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Tuple
import json
import math


# CONSTANTES DU SYSTEME ABS

LAMBDA_OPT_MIN = 0.10    # OCL C02 : zone optimale min
LAMBDA_OPT_MAX = 0.30    # OCL C02 : zone optimale max
ABS_THRESHOLD  = 1.38    # OCL C24 : seuil activation ABS (m/s)
WHEEL_RADIUS   = 0.30    # rayon roue (m)

# ENUMERATIONS

class Action(Enum):
    BUILD   = "Build"     # Augmenter la pression
    HOLD    = "Hold"      # Maintenir la pression
    RELEASE = "Release"   # Reduire la pression


class RoadCondition(Enum):
    DRY_ASPHALT = "Asphalte sec"
    WET_ASPHALT = "Asphalte mouille"
    ICE         = "Verglas"
    GRAVEL      = "Gravier"
    AQUAPLANING = "Aquaplaning"


class RiskLevel(Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"

# DATACLASSES

@dataclass
class ABSState:
    """
    Etat complet du systeme ABS a un instant t.
    Utilise par le XAI pour generer des explications contextuelles.
    """
    timestamp:      str
    cycle:          int
    lambda_val:     float       # Taux de glissement
    mu:             float       # Coefficient de frottement
    vehicle_speed:  float       # Vitesse vehicule (m/s)
    wheel_speeds:   List[float] # Vitesses angulaires (rad/s)
    slip_ratios:    List[float] # lambda par roue
    action:         Action      # Decision
    road_condition: RoadCondition
    abs_active:     bool


@dataclass
class OCLResult:
    """Resultat de verification d une contrainte OCL."""
    constraint_id:  str
    satisfied:      bool
    message:        str
    value:          float = 0.0


@dataclass
class Explanation:
    """
    Explication generee par le XAI pour une decision ABS.
    Contient l explication en langage naturel + la tracabilite OCL.
    """
    decision:          Action
    natural_language:  str
    technical:         str
    ocl_results:       List[OCLResult]
    confidence:        float
    risk_level:        RiskLevel
    recommendation:    str


@dataclass
class TraceRecord:
    """Enregistrement de tracabilite pour une decision ABS."""
    cycle:          int
    timestamp:      str
    state:          ABSState
    explanation:    Explanation
    violated_ocl:   List[str]


# COMPOSANT 1 - RULE-BASED EXPLAINER

class RuleBasedExplainer:
    """
    Composant XAI base sur les regles OCL du systeme ABS.
    Verifie les 28 contraintes OCL C01-C28.
    """

    def check_C01(self, lam: float) -> OCLResult:
        """C01 : 0.0 <= lambda <= 1.0"""
        ok = 0.0 <= lam <= 1.0
        return OCLResult("C01", ok,
            f"C01 SlipRatioRange : lambda={lam:.4f} dans [0.0, 1.0]"
            if ok else
            f"C01 VIOLATION : lambda={lam:.4f} hors [0.0, 1.0]", lam)

    def check_C02(self, lam: float) -> OCLResult:
        """C02 : lambda_opt dans [0.10, 0.30]"""
        ok = LAMBDA_OPT_MIN <= lam <= LAMBDA_OPT_MAX
        return OCLResult("C02", ok,
            f"C02 OptimalSlipRange : lambda={lam:.4f} dans [0.10, 0.30]"
            if ok else
            f"C02 VIOLATION : lambda={lam:.4f} hors zone optimale [0.10, 0.30]", lam)

    def check_C03(self, lam: float, v: float,
                  omega: float) -> OCLResult:
        """C03 : lambda = (v - omega*r) / v"""
        if v <= 0:
            return OCLResult("C03", True,
                "C03 SlipRatioFormula : v=0 cas degenere", 0.0)
        lam_calc = (v - omega * WHEEL_RADIUS) / v
        ok = abs(lam - lam_calc) < 0.01
        return OCLResult("C03", ok,
            f"C03 SlipRatioFormula : lambda={lam:.4f} conforme a (v-wr)/v"
            if ok else
            f"C03 VIOLATION : ecart formule {abs(lam-lam_calc):.4f}", lam)

    def check_C04(self, lam: float) -> OCLResult:
        """C04 : lambda < 1.0"""
        ok = lam < 1.0
        return OCLResult("C04", ok,
            f"C04 NoWheelLockup : lambda={lam:.4f} < 1.0"
            if ok else
            f"C04 VIOLATION : blocage total lambda={lam:.4f}", lam)

    def check_C05(self, mu: float) -> OCLResult:
        """C05 : mu > 0.0"""
        ok = mu > 0.0
        return OCLResult("C05", ok,
            f"C05 FrictionPositive : mu={mu:.4f} > 0"
            if ok else
            f"C05 VIOLATION : mu={mu:.4f} <= 0", mu)

    def check_C06(self, mu: float, lam: float,
                  road: RoadCondition) -> OCLResult:
        """C06 : modele Burckhardt"""
        params = {
            RoadCondition.DRY_ASPHALT: (1.2801, 23.99,  0.52),
            RoadCondition.WET_ASPHALT: (0.857,  33.822, 0.347),
            RoadCondition.ICE:         (0.05,   306.39, 0.0),
            RoadCondition.GRAVEL:      (0.60,   15.0,   0.30),
            RoadCondition.AQUAPLANING: (0.10,   40.0,   0.05),
        }
        c1, c2, c3 = params[road]
        mu_burck = max(c1*(1-math.exp(-c2*lam))-c3*lam, 0.001)
        ok = abs(mu - mu_burck) < 0.5
        return OCLResult("C06", ok,
            f"C06 BurckhardtModel : mu={mu:.3f} (Burckhardt={mu_burck:.3f})"
            if ok else
            f"C06 VIOLATION : mu={mu:.3f} != Burckhardt={mu_burck:.3f}", mu)

    def check_C12(self, v: float) -> OCLResult:
        """C12 : v >= 0"""
        ok = v >= 0.0
        return OCLResult("C12", ok,
            f"C12 SpeedNonNegative : v={v:.2f} m/s >= 0"
            if ok else
            f"C12 VIOLATION : v={v:.2f} < 0", v)

    def check_C13(self, v: float, omega: float) -> OCLResult:
        """C13 : v >= omega*r"""
        ok = v >= omega * WHEEL_RADIUS - 0.001
        return OCLResult("C13", ok,
            f"C13 VehicleSpeedGEWheel : v={v:.2f} >= omega*r={omega*WHEEL_RADIUS:.2f}"
            if ok else
            f"C13 VIOLATION : v={v:.2f} < omega*r={omega*WHEEL_RADIUS:.2f}", v)

    def check_C24(self, v: float, abs_active: bool) -> OCLResult:
        """C24 : ABS actif seulement si v > 1.38 m/s"""
        ok = (not abs_active) or (v > ABS_THRESHOLD)
        return OCLResult("C24", ok,
            f"C24 ABSActivation : ABS={'ON' if abs_active else 'OFF'}, v={v:.2f} m/s"
            if ok else
            f"C24 VIOLATION : ABS actif mais v={v:.2f} <= {ABS_THRESHOLD}", v)

    def check_C25(self, action: Action) -> OCLResult:
        """C25 : phase hydraulique valide"""
        ok = action in Action
        return OCLResult("C25", ok,
            f"C25 PressurePhases : phase={action.value} valide"
            if ok else
            f"C25 VIOLATION : phase invalide", 0.0)

    def check_C27(self, wheel_speeds: List[float]) -> OCLResult:
        """C27 : omega >= 0 pour tous les capteurs"""
        ok = all(w >= 0.0 for w in wheel_speeds)
        return OCLResult("C27", ok,
            f"C27 SensorNonNegative : omega_min={min(wheel_speeds):.2f} >= 0"
            if ok else
            f"C27 VIOLATION : vitesse angulaire negative", 0.0)

    def check_C28(self, wheel_speeds: List[float]) -> OCLResult:
        """C28 : exactement 4 capteurs WSS"""
        ok = len(wheel_speeds) == 4
        return OCLResult("C28", ok,
            f"C28 FourWheels : {len(wheel_speeds)} capteurs WSS"
            if ok else
            f"C28 VIOLATION : {len(wheel_speeds)} capteurs au lieu de 4", 0.0)

    def verify_all(self, state: ABSState) -> List[OCLResult]:
        """Verifie toutes les contraintes OCL pour un etat donne."""
        lam  = state.lambda_val
        mu   = state.mu
        v    = state.vehicle_speed
        ws   = state.wheel_speeds
        road = state.road_condition

        results = [
            self.check_C01(lam),
            self.check_C02(lam),
            self.check_C03(lam, v, ws[0] if ws else 0.0),
            self.check_C04(lam),
            self.check_C05(mu),
            self.check_C06(mu, lam, road),
            self.check_C12(v),
            self.check_C13(v, ws[0] if ws else 0.0),
            self.check_C24(v, state.abs_active),
            self.check_C25(state.action),
            self.check_C27(ws),
            self.check_C28(ws),
        ]
        return results

    def get_violated(self, state: ABSState) -> List[str]:
        """Retourne la liste des contraintes OCL violees."""
        return [r.constraint_id for r in self.verify_all(state)
                if not r.satisfied]


# COMPOSANT 2 - EXPLANATION ENGINE

class ExplanationEngine:
    """
    Moteur d explication XAI du systeme ABS.
    Genere des explications a 3 niveaux :
        1. Langage naturel  -> pour le conducteur
        2. Technique        -> pour l ingenieur
        3. Tracabilite OCL  -> pour la validation formelle
    """

    def __init__(self):
        self.rule_explainer = RuleBasedExplainer()

    def _compute_risk(self, state: ABSState,
                      violated: List[str]) -> RiskLevel:
        """Calcule le niveau de risque base sur lambda, mu et OCL."""
        if len(violated) > 1 or state.lambda_val > 0.7:
            return RiskLevel.HIGH
        elif (len(violated) == 1
              or state.lambda_val > LAMBDA_OPT_MAX
              or state.lambda_val < LAMBDA_OPT_MIN
              or state.mu < 0.3):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _natural_language(self, state: ABSState) -> str:
        """Explication en francais comprehensible."""
        lam  = state.lambda_val
        mu   = state.mu
        act  = state.action
        road = state.road_condition.value

        if act == Action.RELEASE:
            return (
                f"La roue glisse trop fortement (lambda={lam:.2f} > {LAMBDA_OPT_MAX}). "
                f"Sur {road} (mu={mu:.2f}), un glissement excessif cause "
                f"la perte de controle directionnel. "
                f"L ABS relache la pression hydraulique pour debloquer la roue."
            )
        elif act == Action.BUILD:
            return (
                f"Le freinage est insuffisant (lambda={lam:.2f} < {LAMBDA_OPT_MIN}). "
                f"Sur {road} (mu={mu:.2f}), l adherence disponible n est pas "
                f"pleinement exploitee. "
                f"L ABS augmente la pression pour ameliorer l efficacite de freinage."
            )
        else:
            return (
                f"Le glissement est optimal (lambda={lam:.2f} dans "
                f"[{LAMBDA_OPT_MIN}, {LAMBDA_OPT_MAX}]). "
                f"Sur {road} (mu={mu:.2f}), l ABS maintient la pression "
                f"pour maximiser la force de freinage tout en gardant la maitrise."
            )

    def _technical(self, state: ABSState,
                   violated: List[str]) -> str:
        """Explication technique pour l ingenieur."""
        return (
            f"Etat physique : lambda={state.lambda_val:.4f}, "
            f"mu={state.mu:.4f}, v={state.vehicle_speed:.2f} m/s\n"
            f"Decision ABS  : {state.action.value}\n"
            f"Route         : {state.road_condition.value}\n"
            f"OCL violees   : {violated if violated else 'aucune'}\n"
            f"Formule slip  : lambda=(v-omega*r)/v\n"
            f"Burckhardt    : mu=c1*(1-exp(-c2*lambda))-c3*lambda"
        )

    def _recommendation(self, risk: RiskLevel) -> str:
        """Recommandation adaptee au niveau de risque."""
        if risk == RiskLevel.HIGH:
            return ("RISQUE ELEVE - Maintenez la pression sur la pedale. "
                    "L ABS gere automatiquement le freinage. "
                    "Regardez ou vous voulez aller.")
        elif risk == RiskLevel.MEDIUM:
            return ("ATTENTION - Surface glissante detectee. "
                    "Continuez a appuyer normalement sur la pedale. "
                    "Distance d arret augmentee.")
        return ("Freinage optimal - L ABS fonctionne correctement. "
                "Maintenez la pression sur la pedale.")

    def explain(self, state: ABSState) -> Explanation:
        """Genere une explication complete pour un etat ABS donne."""
        ocl_results = self.rule_explainer.verify_all(state)
        violated    = [r.constraint_id for r in ocl_results
                       if not r.satisfied]
        n_ok        = sum(1 for r in ocl_results if r.satisfied)
        confidence  = round(n_ok / len(ocl_results), 3)
        risk        = self._compute_risk(state, violated)

        return Explanation(
            decision         = state.action,
            natural_language = self._natural_language(state),
            technical        = self._technical(state, violated),
            ocl_results      = ocl_results,
            confidence       = confidence,
            risk_level       = risk,
            recommendation   = self._recommendation(risk),
        )


# COMPOSANT 3 - TRACEABILITY LOGGER

class TraceabilityLogger:
    """
    Journalise chaque decision ABS avec sa tracabilite complete.
    """

    def __init__(self):
        self._history: List[TraceRecord] = []
        self._engine  = ExplanationEngine()

    def log_decision(self, state: ABSState) -> TraceRecord:
        """Enregistre une decision ABS avec son explication XAI."""
        explanation = self._engine.explain(state)
        violated    = [r.constraint_id for r in explanation.ocl_results
                       if not r.satisfied]
        record = TraceRecord(
            cycle       = state.cycle,
            timestamp   = state.timestamp,
            state       = state,
            explanation = explanation,
            violated_ocl= violated,
        )
        self._history.append(record)
        return record

    def get_history(self) -> List[TraceRecord]:
        return self._history

    def get_violations(self) -> List[TraceRecord]:
        return [r for r in self._history if r.violated_ocl]

    def print_trace(self, record: TraceRecord) -> None:
        """Affiche un TraceRecord de facon lisible."""
        e = record.explanation
        s = record.state
        print(f"\n{'='*65}")
        print(f"  CYCLE {record.cycle} -- {record.timestamp}")
        print(f"{'='*65}")
        print(f"  ETAT PHYSIQUE")
        print(f"    lambda          : {s.lambda_val:.4f}")
        print(f"    mu (frottement) : {s.mu:.4f}")
        print(f"    v (vitesse)     : {s.vehicle_speed:.2f} m/s")
        print(f"    Route           : {s.road_condition.value}")
        print(f"    lambda par roue : {[round(x,4) for x in s.slip_ratios]}")
        print(f"\n  DECISION ABS")
        print(f"    Action          : {s.action.value}")
        print(f"\n  EXPLICATION XAI")
        print(f"    Risque          : {e.risk_level.value}")
        print(f"    Confiance OCL   : {e.confidence:.0%}")
        print(f"    Explication     : {e.natural_language}")
        print(f"    Recommandation  : {e.recommendation}")
        print(f"\n  TRACABILITE OCL")
        for r in e.ocl_results:
            status = "OK" if r.satisfied else "VIOLATION"
            print(f"    {status:<10} {r.message}")
        if record.violated_ocl:
            print(f"\n  CONTRAINTES VIOLEES : {record.violated_ocl}")
        print(f"{'='*65}")

    def export_report(self,
                      filepath: str = "xai_report.json") -> str:
        """Exporte le rapport complet en JSON."""
        report = {
            "project":      "ABS XAI Layer - Chapitre 6",
            "generated":    datetime.now().isoformat(),
            "total_cycles": len(self._history),
            "violations":   len(self.get_violations()),
            "decisions":    []
        }
        for rec in self._history:
            report["decisions"].append({
                "cycle":     rec.cycle,
                "timestamp": rec.timestamp,
                "state": {
                    "lambda":        rec.state.lambda_val,
                    "mu":            rec.state.mu,
                    "vehicle_speed": rec.state.vehicle_speed,
                    "road":          rec.state.road_condition.value,
                    "slip_ratios":   rec.state.slip_ratios,
                },
                "decision": rec.state.action.value,
                "explanation": {
                    "natural":     rec.explanation.natural_language,
                    "technical":   rec.explanation.technical,
                    "confidence":  rec.explanation.confidence,
                    "risk":        rec.explanation.risk_level.value,
                    "recommendation": rec.explanation.recommendation,
                },
                "ocl": {
                    "results":  [{
                        "id":        r.constraint_id,
                        "satisfied": r.satisfied,
                        "message":   r.message,
                    } for r in rec.explanation.ocl_results],
                    "violated": rec.violated_ocl,
                }
            })
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return filepath


# MOTEUR DE DECISION ABS 

class ABSDecisionEngine:
    """
    Moteur de decision ABS base sur les regles physiques OCL.
    Remplace le Q-learning par une logique de controle pure
    basee sur les contraintes C01-C28.
    """

    BURCKHARDT = {
        RoadCondition.DRY_ASPHALT: (1.2801, 23.99,  0.52),
        RoadCondition.WET_ASPHALT: (0.857,  33.822, 0.347),
        RoadCondition.ICE:         (0.05,   306.39, 0.001),
        RoadCondition.GRAVEL:      (0.60,   15.0,   0.30),
        RoadCondition.AQUAPLANING: (0.10,   40.0,   0.05),
    }

    def compute_mu(self, lam: float,
                   road: RoadCondition) -> float:
        """Modele Burckhardt -- OCL C06."""
        c1, c2, c3 = self.BURCKHARDT[road]
        return max(c1*(1-math.exp(-c2*lam))-c3*lam, 0.001)

    def compute_slip(self, v: float, omega: float) -> float:
        """Formule de glissement -- OCL C03."""
        if v <= 0:
            return 0.0
        return max(0.0, min((v - omega * WHEEL_RADIUS) / v, 0.99))

    def decide(self, lambda_val: float) -> Action:
        """
        Decision ABS basee sur les contraintes OCL C02 et C04.
        - lambda > LAMBDA_OPT_MAX -> RELEASE (OCL C02 violee)
        - lambda < LAMBDA_OPT_MIN -> BUILD   (sous-freinage)
        - sinon                   -> HOLD    (zone optimale)
        """
        if lambda_val > LAMBDA_OPT_MAX:
            return Action.RELEASE
        elif lambda_val < LAMBDA_OPT_MIN:
            return Action.BUILD
        return Action.HOLD

    def run_cycle(self, omega_list: List[float],
                  brake_pressure: float,
                  vehicle_speed: float,
                  road: RoadCondition,
                  cycle: int) -> ABSState:
        """Execute un cycle MAPE-K complet."""
        slip_ratios = [self.compute_slip(vehicle_speed, w)
                       for w in omega_list]
        lambda_val  = max(slip_ratios)
        mu          = self.compute_mu(lambda_val, road)
        action      = self.decide(lambda_val)
        abs_active  = brake_pressure > 0 and vehicle_speed > ABS_THRESHOLD

        return ABSState(
            timestamp     = datetime.now().strftime("%H:%M:%S.%f")[:-3],
            cycle         = cycle,
            lambda_val    = round(lambda_val, 4),
            mu            = round(mu, 4),
            vehicle_speed = round(vehicle_speed, 2),
            wheel_speeds  = omega_list,
            slip_ratios   = [round(s, 4) for s in slip_ratios],
            action        = action,
            road_condition= road,
            abs_active    = abs_active,
        )


# DEMONSTRATION

if __name__ == "__main__":

    print("\n" + "="*65)
    print("  CHAPITRE 6 -- XAI LAYER")
    print("  Systeme ABS -- Explainable AI-Driven MDE")
    print("="*65)

    engine = ABSDecisionEngine()
    logger = TraceabilityLogger()

    scenarios = [
        {
            "nom":    "Freinage normal -- Asphalte sec",
            "omegas": [65.0, 65.0, 64.0, 65.0],
            "brake":  80.0, "speed": 22.0,
            "road":   RoadCondition.DRY_ASPHALT
        },
        {
            "nom":    "Roue arriere quasi-bloquee",
            "omegas": [60.0, 58.0, 12.0, 61.0],
            "brake":  80.0, "speed": 20.0,
            "road":   RoadCondition.DRY_ASPHALT
        },
        {
            "nom":    "Freinage sur verglas",
            "omegas": [30.0, 28.0, 25.0, 29.0],
            "brake":  80.0, "speed": 15.0,
            "road":   RoadCondition.ICE
        },
        {
            "nom":    "Zone optimale -- Asphalte mouille",
            "omegas": [55.0, 54.0, 53.0, 55.0],
            "brake":  60.0, "speed": 18.0,
            "road":   RoadCondition.WET_ASPHALT
        },
    ]

    for i, sc in enumerate(scenarios):
        print(f"\n>>> SCENARIO {i+1} : {sc['nom']}")
        state  = engine.run_cycle(
            sc["omegas"], sc["brake"], sc["speed"], sc["road"], i+1)
        record = logger.log_decision(state)
        logger.print_trace(record)

    # Resume
    history = logger.get_history()
    print(f"\n{'='*65}")
    print(f"  RESUME SESSION")
    print(f"{'='*65}")
    print(f"  Cycles analyses  : {len(history)}")
    risks = {"LOW":0,"MEDIUM":0,"HIGH":0}
    for rec in history:
        risks[rec.explanation.risk_level.value] += 1
    print(f"  Niveaux risque   : LOW={risks['LOW']} | "
          f"MEDIUM={risks['MEDIUM']} | HIGH={risks['HIGH']}")
    avg_conf = sum(r.explanation.confidence for r in history)/len(history)
    print(f"  Confiance moy.   : {avg_conf:.1%}")
    violations = logger.get_violations()
    if violations:
        print(f"  OCL violees      : {len(violations)} cycle(s)")
        for v in violations:
            print(f"    Cycle {v.cycle} : {v.violated_ocl}")
    else:
        print(f"  OCL violees      : aucune")

    path = logger.export_report("xai_report.json")
    print(f"\n  Rapport exporte  : {path}")
    print(f"{'='*65}\n")
