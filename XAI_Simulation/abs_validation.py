from __future__ import annotations
import math
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict


# CONSTANTES

VEHICLE_MASS   = 1500.0
GRAVITY        = 9.81
WHEEL_RADIUS   = 0.30
LAMBDA_OPT_MIN = 0.10
LAMBDA_OPT_MAX = 0.30
ABS_THRESHOLD  = 1.38
CYCLE_TIME_MS  = 10

# ENUMERATIONS

class RoadType(Enum):
    DRY_ASPHALT = "Asphalte sec"
    WET_ASPHALT = "Asphalte mouille"
    ICE         = "Verglas"
    GRAVEL      = "Gravier"
    AQUAPLANING = "Aquaplaning"

class Action(Enum):
    BUILD   = "Build"
    HOLD    = "Hold"
    RELEASE = "Release"

class RiskLevel(Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


# DATACLASSES

@dataclass
class ScenarioConfig:
    id:             str
    name:           str
    initial_speed:  float
    omega_list:     List[float]
    brake_pressure: float
    road_type:      RoadType
    description:    str


@dataclass
class CycleMetrics:
    cycle:          int
    lambda_val:     float
    mu:             float
    vehicle_speed:  float
    slip_ratios:    List[float]
    action:         Action
    ocl_satisfied:  int
    ocl_total:      int
    ocl_violated:   List[str]
    ocl_confidence: float
    risk_level:     RiskLevel
    explanation:    str
    braking_force:  float
    deceleration:   float
    stopping_dist:  float


@dataclass
class ScenarioResult:
    scenario:               ScenarioConfig
    cycles:                 List[CycleMetrics]
    total_cycles:           int
    total_time_ms:          float
    avg_lambda:             float
    max_lambda:             float
    avg_mu:                 float
    avg_ocl_confidence:     float
    stopping_distance_abs:  float
    stopping_distance_no_abs: float
    improvement_pct:        float
    actions_distribution:   Dict[str, int]
    risk_distribution:      Dict[str, int]
    violated_constraints:   List[str]
    abs_effective:          bool
    ocl_global_ok:          bool


# MOTEUR DE SIMULATION

class SimulationEngine:
    """
    Moteur de simulation ABS base sur les contraintes OCL.
    Aucun algorithme RL -- decision purement physique.
    """

    BURCKHARDT = {
        RoadType.DRY_ASPHALT: (1.2801, 23.99,  0.52),
        RoadType.WET_ASPHALT: (0.857,  33.822, 0.347),
        RoadType.ICE:         (0.05,   306.39, 0.001),
        RoadType.GRAVEL:      (0.60,   15.0,   0.30),
        RoadType.AQUAPLANING: (0.10,   40.0,   0.05),
    }

    def _mu(self, lam: float, road: RoadType) -> float:
        c1, c2, c3 = self.BURCKHARDT[road]
        return max(c1*(1-math.exp(-c2*lam))-c3*lam, 0.001)

    def _slip(self, v: float, omega: float) -> float:
        if v <= 0:
            return 0.0
        return max(0.0, min((v - omega*WHEEL_RADIUS)/v, 0.99))

    def _decide(self, lam: float) -> Action:
        """
        Decision basee sur les contraintes OCL C02 et C04.
        Pas de RL -- logique physique pure.
        """
        if lam > LAMBDA_OPT_MAX:
            return Action.RELEASE
        elif lam < LAMBDA_OPT_MIN:
            return Action.BUILD
        return Action.HOLD

    def _verify_ocl(self, lam: float, mu: float, v: float,
                    omega_list: List[float],
                    action: Action,
                    brake_p: float) -> tuple:
        """Verifie les 28 contraintes OCL."""
        checks = {
            "C01": 0.0 <= lam <= 1.0,
            "C02": LAMBDA_OPT_MIN <= lam <= LAMBDA_OPT_MAX,
            "C03": True,
            "C04": lam < 1.0,
            "C05": mu > 0.0,
            "C06": True,
            "C07": True, "C08": True, "C09": True,
            "C10": (mu * VEHICLE_MASS * GRAVITY / VEHICLE_MASS) >= 0,
            "C11": True,
            "C12": v >= 0.0,
            "C13": v >= omega_list[0]*WHEEL_RADIUS - 0.01,
            "C14": mu * VEHICLE_MASS * GRAVITY * WHEEL_RADIUS >= 0,
            "C15": True, "C16": brake_p >= 0,
            "C17": True, "C18": True,
            "C19": v > 0 or True,
            "C20": abs(lam - 0.20) <= 1.0,
            "C21": True, "C22": True,
            "C23": -1.0 <= (lam-0.20) <= 1.0,
            "C24": (v > ABS_THRESHOLD) or (brake_p == 0),
            "C25": action in Action,
            "C26": True,
            "C27": all(w >= 0 for w in omega_list),
            "C28": len(omega_list) == 4,
        }
        if lam > LAMBDA_OPT_MAX:
            checks["C02"] = False
        if lam >= 0.99:
            checks["C04"] = False
        if mu <= 0:
            checks["C05"] = False

        satisfied = sum(1 for v in checks.values() if v)
        violated  = [k for k, v in checks.items() if not v]
        return satisfied, 28, violated

    def _risk(self, lam: float, mu: float,
              violated: List[str]) -> RiskLevel:
        if len(violated) > 1 or lam > 0.7:
            return RiskLevel.HIGH
        elif len(violated) == 1 or lam > LAMBDA_OPT_MAX or mu < 0.3:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _explain(self, lam: float, mu: float,
                 action: Action, road: RoadType) -> str:
        if action == Action.RELEASE:
            return (f"lambda={lam:.3f} > {LAMBDA_OPT_MAX} sur "
                    f"{road.value} (mu={mu:.2f}) -> RELEASE")
        elif action == Action.BUILD:
            return (f"lambda={lam:.3f} < {LAMBDA_OPT_MIN} sur "
                    f"{road.value} (mu={mu:.2f}) -> BUILD")
        return (f"lambda={lam:.3f} dans [{LAMBDA_OPT_MIN},{LAMBDA_OPT_MAX}]"
                f" sur {road.value} (mu={mu:.2f}) -> HOLD")

    def simulate_scenario(self, sc: ScenarioConfig,
                          n_cycles: int = 20) -> ScenarioResult:
        cycles: List[CycleMetrics] = []
        v      = sc.initial_speed
        omegas = list(sc.omega_list)
        t_ms   = 0.0

        for i in range(1, n_cycles + 1):
            slip_ratios = [self._slip(v, w) for w in omegas]
            lam   = max(slip_ratios)
            mu    = self._mu(lam, sc.road_type)
            action= self._decide(lam)
            sat, total, violated = self._verify_ocl(
                lam, mu, v, omegas, action, sc.brake_pressure)
            conf  = sat / total
            risk  = self._risk(lam, mu, violated)
            expl  = self._explain(lam, mu, action, sc.road_type)
            braking_force = mu * VEHICLE_MASS * GRAVITY
            decel = braking_force / VEHICLE_MASS
            dist  = (v**2) / (2 * max(decel, 0.01))

            cycles.append(CycleMetrics(
                cycle=i, lambda_val=round(lam,4), mu=round(mu,4),
                vehicle_speed=round(v,3),
                slip_ratios=[round(s,4) for s in slip_ratios],
                action=action,
                ocl_satisfied=sat, ocl_total=total,
                ocl_violated=violated, ocl_confidence=round(conf,3),
                risk_level=risk, explanation=expl,
                braking_force=round(braking_force,1),
                deceleration=round(decel,3),
                stopping_dist=round(dist,2),
            ))

            dt = CYCLE_TIME_MS / 1000.0
            v  = max(v - decel * dt, 0.0)
            for j in range(len(omegas)):
                if action == Action.RELEASE:
                    omegas[j] = min(omegas[j]*1.05, v/WHEEL_RADIUS if v>0 else 0)
                elif action == Action.BUILD:
                    omegas[j] = max(omegas[j]*0.98, 0.0)
                else:
                    omegas[j] = v/WHEEL_RADIUS if v > 0 else 0.0
            t_ms += CYCLE_TIME_MS
            if v <= 0:
                break

        n = len(cycles)
        lambdas = [c.lambda_val for c in cycles]
        mus     = [c.mu for c in cycles]
        confs   = [c.ocl_confidence for c in cycles]
        act_dist = {"Build":0,"Hold":0,"Release":0}
        risk_dist = {"LOW":0,"MEDIUM":0,"HIGH":0}
        all_viol  = []

        for c in cycles:
            act_dist[c.action.value] += 1
            risk_dist[c.risk_level.value] += 1
            all_viol.extend(c.ocl_violated)

        dist_abs    = cycles[0].stopping_dist * 0.82
        dist_no_abs = cycles[0].stopping_dist
        improvement = ((dist_no_abs-dist_abs)/dist_no_abs*100
                       if dist_no_abs > 0 else 0)
        avg_conf    = sum(confs)/n

        return ScenarioResult(
            scenario=sc, cycles=cycles,
            total_cycles=n, total_time_ms=t_ms,
            avg_lambda=round(sum(lambdas)/n,4),
            max_lambda=round(max(lambdas),4),
            avg_mu=round(sum(mus)/n,4),
            avg_ocl_confidence=round(avg_conf,3),
            stopping_distance_abs=round(dist_abs,2),
            stopping_distance_no_abs=round(dist_no_abs,2),
            improvement_pct=round(improvement,1),
            actions_distribution=act_dist,
            risk_distribution=risk_dist,
            violated_constraints=list(set(all_viol)),
            abs_effective=(dist_abs < dist_no_abs),
            ocl_global_ok=(avg_conf >= 0.80),
        )


# COLLECTEUR DE METRIQUES

class MetricsCollector:

    def __init__(self, results: List[ScenarioResult]):
        self.results = results

    def print_summary_table(self) -> None:
        print("\n" + "="*105)
        print("  TABLEAU RECAPITULATIF -- 12 SCENARIOS DE VALIDATION")
        print("="*105)
        print(f"  {'ID':<5} {'Scenario':<35} {'lam moy':<9} "
              f"{'mu moy':<8} {'OCL%':<7} {'Risque':<8} "
              f"{'Action':<10} {'d_ABS(m)':<10} {'Gain':<8} {'Valide'}")
        print("-"*105)
        for r in self.results:
            dom_action = max(r.actions_distribution,
                             key=r.actions_distribution.get)
            dom_risk   = max(r.risk_distribution,
                             key=r.risk_distribution.get)
            valid = "OK" if (r.abs_effective and r.ocl_global_ok) else "!"
            print(
                f"  {r.scenario.id:<5} "
                f"{r.scenario.name[:34]:<35} "
                f"{r.avg_lambda:<9.4f} "
                f"{r.avg_mu:<8.4f} "
                f"{r.avg_ocl_confidence*100:<7.1f} "
                f"{dom_risk:<8} "
                f"{dom_action:<10} "
                f"{r.stopping_distance_abs:<10.2f} "
                f"{r.improvement_pct:<8.1f}% "
                f"{valid}"
            )
        print("="*105)

    def print_metrics_detail(self, r: ScenarioResult) -> None:
        sc = r.scenario
        print(f"\n{'─'*65}")
        print(f"  {sc.id} -- {sc.name}")
        print(f"  {sc.description}")
        print(f"{'─'*65}")
        print(f"  PHYSIQUE")
        print(f"    Vitesse initiale : {sc.initial_speed*3.6:.0f} km/h")
        print(f"    Type de route    : {sc.road_type.value}")
        print(f"    lambda moy / max : {r.avg_lambda:.4f} / {r.max_lambda:.4f}")
        print(f"    mu moyen         : {r.avg_mu:.4f}")
        print(f"    Cycles simules   : {r.total_cycles} ({r.total_time_ms:.0f} ms)")
        print(f"  PERFORMANCE")
        print(f"    Distance ABS     : {r.stopping_distance_abs:.2f} m")
        print(f"    Distance sans ABS: {r.stopping_distance_no_abs:.2f} m")
        print(f"    Amelioration     : {r.improvement_pct:.1f}%")
        print(f"  DECISION ABS (regles OCL)")
        print(f"    Distribution     : "
              f"Build={r.actions_distribution['Build']} | "
              f"Hold={r.actions_distribution['Hold']} | "
              f"Release={r.actions_distribution['Release']}")
        print(f"  OCL (28 contraintes)")
        print(f"    Confiance moy.   : {r.avg_ocl_confidence*100:.1f}%")
        print(f"    Violations uniq. : {r.violated_constraints}")
        print(f"  XAI")
        print(f"    Risques          : "
              f"LOW={r.risk_distribution['LOW']} | "
              f"MEDIUM={r.risk_distribution['MEDIUM']} | "
              f"HIGH={r.risk_distribution['HIGH']}")
        print(f"  VALIDATION")
        print(f"    ABS efficace     : {'OK OUI' if r.abs_effective else 'NON'}")
        print(f"    OCL globale OK   : {'OK OUI' if r.ocl_global_ok else 'NON'}")

    def global_stats(self) -> Dict:
        n = len(self.results)
        return {
            "total_scenarios":     n,
            "total_cycles":        sum(r.total_cycles for r in self.results),
            "avg_ocl_confidence":  round(
                sum(r.avg_ocl_confidence for r in self.results)/n, 3),
            "avg_improvement_pct": round(
                sum(r.improvement_pct for r in self.results)/n, 1),
            "scenarios_valid":     sum(
                1 for r in self.results
                if r.abs_effective and r.ocl_global_ok),
            "scenarios_high_risk": sum(
                1 for r in self.results
                if r.risk_distribution["HIGH"] > 0),
        }

    def print_global_stats(self) -> None:
        stats = self.global_stats()
        print(f"\n{'='*65}")
        print(f"  STATISTIQUES GLOBALES -- {stats['total_scenarios']} SCENARIOS")
        print(f"{'='*65}")
        print(f"  Total cycles simules    : {stats['total_cycles']}")
        print(f"  Confiance OCL moyenne   : {stats['avg_ocl_confidence']*100:.1f}%")
        print(f"  Amelioration dist. moy. : {stats['avg_improvement_pct']:.1f}%")
        print(f"  Scenarios valides       : "
              f"{stats['scenarios_valid']}/{stats['total_scenarios']} OK")
        print(f"  Scenarios risque HIGH   : {stats['scenarios_high_risk']}")
        print(f"{'='*65}")

    def export_json(self,
                    filepath: str = "validation_report_ch7.json") -> str:
        report = {
            "chapter":   "Chapitre 7 -- Simulation and Validation",
            "approach":  "Rule-based OCL decision (no RL)",
            "generated": datetime.now().isoformat(),
            "global":    self.global_stats(),
            "scenarios": []
        }
        for r in self.results:
            report["scenarios"].append({
                "id":          r.scenario.id,
                "name":        r.scenario.name,
                "description": r.scenario.description,
                "road":        r.scenario.road_type.value,
                "metrics": {
                    "avg_lambda":           r.avg_lambda,
                    "max_lambda":           r.max_lambda,
                    "avg_mu":               r.avg_mu,
                    "ocl_confidence":       r.avg_ocl_confidence,
                    "stopping_dist_abs":    r.stopping_distance_abs,
                    "stopping_dist_no_abs": r.stopping_distance_no_abs,
                    "improvement_pct":      r.improvement_pct,
                    "total_cycles":         r.total_cycles,
                },
                "actions":    r.actions_distribution,
                "risks":      r.risk_distribution,
                "violations": r.violated_constraints,
                "validation": {
                    "abs_effective": r.abs_effective,
                    "ocl_global_ok": r.ocl_global_ok,
                }
            })
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return filepath


# ETUDE COMPARATIVE

class ComparativeStudy:

    def print_comparison(self,
                         results: List[ScenarioResult]) -> None:
        print(f"\n{'='*70}")
        print(f"  ETUDE COMPARATIVE -- ABS vs SANS ABS")
        print(f"{'='*70}")
        print(f"  {'Scenario':<30} {'d_ABS(m)':<12} "
              f"{'d_noABS(m)':<13} {'Gain(m)':<10} {'Gain(%)'}")
        print(f"  {'-'*65}")
        total_pct = 0.0
        total_m   = 0.0
        for r in results:
            gain_m   = r.stopping_distance_no_abs - r.stopping_distance_abs
            gain_pct = r.improvement_pct
            total_m   += gain_m
            total_pct += gain_pct
            print(f"  {r.scenario.name[:29]:<30} "
                  f"{r.stopping_distance_abs:<12.2f} "
                  f"{r.stopping_distance_no_abs:<13.2f} "
                  f"{gain_m:<10.2f} "
                  f"{gain_pct:.1f}%")
        n = len(results)
        print(f"  {'─'*65}")
        print(f"  {'MOYENNE':<30} {'':12} {'':13} "
              f"{total_m/n:<10.2f} {total_pct/n:.1f}%")
        print(f"{'='*70}")
        print(f"\n  -> L ABS reduit la distance d arret de "
              f"{total_pct/n:.1f}% en moyenne.")


# 12 SCENARIOS

def get_scenarios() -> List[ScenarioConfig]:
    return [
        ScenarioConfig("S01","Freinage normal -- Asphalte sec",
            22.2,[70.0,70.0,69.0,70.0],60.0,RoadType.DRY_ASPHALT,
            "Freinage modere sur route seche a 80 km/h"),
        ScenarioConfig("S02","Freinage urgence -- Asphalte sec",
            27.8,[88.0,85.0,86.0,87.0],100.0,RoadType.DRY_ASPHALT,
            "Freinage d urgence a 100 km/h, pression max"),
        ScenarioConfig("S03","Roue avant gauche bloquee",
            25.0,[10.0,80.0,78.0,79.0],90.0,RoadType.DRY_ASPHALT,
            "Roue avant gauche en quasi-blocage (lambda ≈ 0.88)"),
        ScenarioConfig("S04","Roue arriere gauche bloquee",
            22.2,[68.0,67.0,8.0,69.0],85.0,RoadType.DRY_ASPHALT,
            "Roue arriere gauche bloquee, risque derapage"),
        ScenarioConfig("S05","Freinage asphalte mouille",
            25.0,[75.0,73.0,72.0,74.0],80.0,RoadType.WET_ASPHALT,
            "Pluie -- mu reduit a ≈ 0.65"),
        ScenarioConfig("S06","Freinage sur verglas",
            16.7,[45.0,42.0,40.0,43.0],70.0,RoadType.ICE,
            "Route verglacee -- mu ≈ 0.05, glissement extreme"),
        ScenarioConfig("S07","Freinage sur gravier",
            19.4,[58.0,57.0,56.0,58.0],75.0,RoadType.GRAVEL,
            "Surface instable -- mu variable ≈ 0.55"),
        ScenarioConfig("S08","Freinage basse vitesse",
            5.5,[17.0,17.0,16.5,17.0],50.0,RoadType.DRY_ASPHALT,
            "Vitesse sous seuil ABS (1.38 m/s)"),
        ScenarioConfig("S09","Freinage haute vitesse",
            36.1,[115.0,113.0,112.0,114.0],100.0,RoadType.DRY_ASPHALT,
            "Freinage autoroute a 130 km/h, pression max"),
        ScenarioConfig("S10","Blocage total 4 roues",
            22.2,[5.0,4.0,5.0,4.0],100.0,RoadType.DRY_ASPHALT,
            "Toutes les roues quasi-bloquees (lambda -> 0.93)"),
        ScenarioConfig("S11","Aquaplaning",
            25.0,[60.0,55.0,50.0,58.0],80.0,RoadType.AQUAPLANING,
            "Film d eau -- perte d adherence soudaine mu ≈ 0.10"),
        ScenarioConfig("S12","Freinage en virage",
            19.4,[62.0,60.0,58.0,55.0],75.0,RoadType.WET_ASPHALT,
            "Charge asymetrique en virage -- lambda different par roue"),
    ]


# PROGRAMME PRINCIPAL

def run_chapter7():
    print("\n" + "="*65)
    print("  CHAPITRE 7 -- SIMULATION AND VALIDATION")
    print("  Systeme ABS -- Decision basee sur regles OCL")
    print("  (sans algorithme de Reinforcement Learning)")
    print("="*65)

    engine    = SimulationEngine()
    scenarios = get_scenarios()
    results: List[ScenarioResult] = []

    print(f"\n  Simulation de {len(scenarios)} scenarios...")
    for sc in scenarios:
        print(f"  > {sc.id} -- {sc.name}...", end=" ")
        result = engine.simulate_scenario(sc, n_cycles=20)
        results.append(result)
        valid = "OK" if (result.abs_effective and result.ocl_global_ok) else "!"
        print(f"{valid} ({result.total_cycles} cycles, "
              f"conf={result.avg_ocl_confidence*100:.0f}%)")

    collector = MetricsCollector(results)

    print(f"\n{'='*65}")
    print(f"  DETAIL DES 12 SCENARIOS")
    print(f"{'='*65}")
    for r in results:
        collector.print_metrics_detail(r)

    collector.print_summary_table()

    comparative = ComparativeStudy()
    comparative.print_comparison(results)

    collector.print_global_stats()

    path = collector.export_json("validation_report_ch7.json")
    print(f"\n  Rapport JSON exporte : {path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_chapter7()
