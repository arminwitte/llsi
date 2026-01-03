
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg

from .sysiddata import SysIdData
from .sysidalg import sysid
from .subspace import SubspaceIdent
from .utils import cv
from .ltimodel import LTIModel
from .polynomialmodel import PolynomialModel
from .statespacemodel import StateSpaceModel


@dataclass
class AutoIdentResult:
    """
    Result object of the autoident function.
    Contains the final model, the (possibly preprocessed) data, and a report.
    """
    model: Any                 # The winning model (StateSpace or Polynomial)
    data: Any                  # The used SysIdData object (after resampling/centering)
    metrics: Dict[str, float]  # e.g. {'fit': 87.5}
    report: List[str] = field(default_factory=list) # Log of decision steps

    def summary(self):
        print(f"=== AutoIdent Result ===")
        print(f"Final Model Type: {type(self.model).__name__}")
        print(f"Metrics: {self.metrics}")
        print("Steps taken:")
        for line in self.report:
            print(f" > {line}")


def compute_hankel_sv(data: SysIdData, horizon: int = 20) -> np.ndarray:
    """
    Compute Hankel singular values for order estimation.
    Uses the logic from subspace identification (N4SID/PO-MOESP).
    """
    keys = list(data.series.keys())
    y_names = sorted([k for k in keys if k.startswith("y") or k == "Nu"])
    u_names = sorted([k for k in keys if k.startswith("u") or k == "Re"])
    
    if not y_names:
        y_names = [k for k in keys if k not in u_names]
    
    if 'y' in data.series: y_names = ['y']
    if 'u' in data.series: u_names = ['u']

    y = np.column_stack([data.series[k] for k in y_names])
    u = np.column_stack([data.series[k] for k in u_names])
    
    r = horizon
    
    Y = SubspaceIdent.hankel(y, 2 * r)
    U = SubspaceIdent.hankel(u, 2 * r)
    
    Yp = Y[0:r, :]
    Up = U[0:r, :]
    
    Yf = Y[r : 2 * r, :]
    Uf = U[r : 2 * r, :]
    
    Wp = np.vstack((Up, Yp))
    Psi = np.vstack((np.vstack((Uf, Wp)), Yf))
    
    L, _ = SubspaceIdent.lq(Psi)
    
    L22 = L[r : 3 * r, r : 3 * r]
    L32 = L[3 * r : 4 * r, r : 3 * r]
    
    Gamma_r = L32 @ np.linalg.pinv(L22) @ Wp
    
    s = scipy.linalg.svd(Gamma_r, compute_uv=False)
    
    return s


def find_gaps_in_sv(sv: np.ndarray, top_k: int = 3) -> List[int]:
    """
    Find the most likely model orders based on gaps in singular values.
    """
    sv = sv[sv > 1e-10]
    if len(sv) < 2:
        return [1]
        
    ratios = sv[:-1] / sv[1:]
    sorted_indices = np.argsort(ratios)[::-1]
    best_indices = sorted_indices[:top_k]
    orders = sorted([int(i) + 1 for i in best_indices])
    
    return orders


def get_data_matrix(data: SysIdData, names: Union[str, List[str]]) -> np.ndarray:
    if isinstance(names, str):
        return data[names].reshape(-1, 1)
    else:
        return np.column_stack([data[n] for n in names])


def ridge_arx(data: SysIdData, order: int, optimize_lambda: bool = True) -> LTIModel:
    """
    Helper to train Ridge ARX model, optionally optimizing lambda.
    """
    keys = list(data.series.keys())
    y_names = sorted([k for k in keys if k.startswith("y") or k == "Nu"])
    u_names = sorted([k for k in keys if k.startswith("u") or k == "Re"])
    if not y_names and 'y' in data.series: y_names = ['y']
    if not u_names and 'u' in data.series: u_names = ['u']
    
    arx_order = (order, order, 0)
    
    best_lambda = 0.0
    
    if optimize_lambda:
        train, val = data.split(0.7)
        
        y_name_cv = y_names[0] if isinstance(y_names, list) else y_names
        u_name_cv = u_names[0] if isinstance(u_names, list) else u_names

        best_lambda, _ = cv(
            train, val, 
            y_name=y_name_cv, u_name=u_name_cv, 
            order=arx_order, 
            method='arx', 
            bounds=(1e-6, 1e2)
        )
    
    mod = sysid(
        data, 
        y_names, 
        u_names, 
        arx_order, 
        method='arx', 
        settings={'lambda': best_lambda}
    )
    
    return mod


def autoident(
    u: np.ndarray, 
    y: np.ndarray, 
    t: Optional[np.ndarray] = None, 
    Ts: Optional[float] = None, 
    max_freq: Optional[float] = None, 
    order_hint: Optional[int] = None, 
    result_type: str = 'state_space',  # 'state_space' or 'polynomial'
    effort: str = 'thorough'           # 'fast' or 'thorough'
) -> AutoIdentResult:
    
    report = []
    report.append("--- Start AutoIdent ---")

    # ---------------------------------------------------------
    # 1. SETUP & PREPROCESSING
    # ---------------------------------------------------------
    
    if t is None and Ts is None:
        raise ValueError("Please provide 't' (time vector) or 'Ts' (sampling time).")
    
    u = np.asarray(u)
    y = np.asarray(y)
    if u.ndim == 1: u = u.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)
    
    series = {}
    for i in range(u.shape[1]):
        series[f"u{i+1}" if u.shape[1] > 1 else "u"] = u[:, i]
    for i in range(y.shape[1]):
        series[f"y{i+1}" if y.shape[1] > 1 else "y"] = y[:, i]
        
    data = SysIdData(t=t, Ts=Ts, **series)
    
    data.center()
    report.append("Data centered (mean removed).")

    if max_freq is not None:
        if data.Ts is None:
             pass
             
        fs_current = 1.0 / data.Ts
        fs_target = 2.5 * max_freq
        
        if fs_current > fs_target:
            factor = int(np.floor(fs_current / fs_target))
            if factor > 1:
                data = data.downsample(q=factor)
                report.append(f"Downsampled data by factor {factor} (New Fs: {1.0/data.Ts:.1f} Hz).")

    train_data, val_data = data.split(proportion=0.7)


    # ---------------------------------------------------------
    # 2. ORDER ESTIMATION
    # ---------------------------------------------------------
    candidate_orders = []

    if order_hint is not None:
        candidate_orders = sorted(list(set([max(1, order_hint-1), order_hint, order_hint+1])))
        report.append(f"Using order hint provided: checking {candidate_orders}")
    else:
        try:
            sv_values = compute_hankel_sv(train_data, horizon=20) 
            candidate_orders = find_gaps_in_sv(sv_values, top_k=3)
            report.append(f"SVD Analysis suggested orders: {candidate_orders}")
        except Exception as e:
            report.append(f"SVD Analysis failed ({e}). Defaulting to orders [1, 2, 3, 4, 5].")
            candidate_orders = [1, 2, 3, 4, 5]


    # ---------------------------------------------------------
    # 3. THE TOURNAMENT (Subspace vs. ARX)
    # ---------------------------------------------------------
    
    best_model = None
    best_score = -np.inf 
    winning_info = ""
    
    keys = list(data.series.keys())
    y_names_list = sorted([k for k in keys if k.startswith("y")])
    u_names_list = sorted([k for k in keys if k.startswith("u")])
    
    y_val = get_data_matrix(val_data, y_names_list)
    u_val = get_data_matrix(val_data, u_names_list)

    for n in candidate_orders:
        # --- Challenger A: Subspace (N4SID) ---
        try:
            mod_sub = sysid(train_data, y_names_list, u_names_list, order=n, method='n4sid')
            score_sub = mod_sub.compare(y_val, u_val)
        except Exception as e:
            report.append(f"N4SID order {n} failed: {e}")
            mod_sub, score_sub = None, -np.inf

        # --- Challenger B: Ridge ARX ---
        try:
            mod_arx = ridge_arx(train_data, order=n, optimize_lambda=True)
            score_arx = mod_arx.compare(y_val, u_val)
        except Exception as e:
            report.append(f"ARX order {n} failed: {e}")
            mod_arx, score_arx = None, -np.inf

        # --- Round Winner ---
        if score_arx > score_sub:
            round_winner = mod_arx
            round_score = score_arx
            source = "RidgeARX"
        else:
            round_winner = mod_sub
            round_score = score_sub
            source = "N4SID"

        # --- Global Winner Update ---
        if round_score > best_score:
            best_score = round_score
            best_model = round_winner
            winning_info = f"{source} (Order {n})"

    report.append(f"Tournament Winner: {winning_info} with Val-Fit: {best_score:.2f}%")


    # ---------------------------------------------------------
    # 4. PEM REFINEMENT (Optional)
    # ---------------------------------------------------------
    final_model = best_model
    final_score = best_score

    if effort in ['thorough', 'pem'] and best_model is not None:
        report.append("Starting PEM refinement...")
        
        try:
            init_method = 'arx' if isinstance(best_model, PolynomialModel) else 'n4sid'
            pem_settings = {'init': init_method}
            
            mod_pem = sysid(train_data, y_names_list, u_names_list, order=n, method='pem', settings=pem_settings)
            
            is_stable = True
            if hasattr(mod_pem, 'is_stable'):
                is_stable = mod_pem.is_stable()
            elif hasattr(mod_pem, 'A'):
                evals = np.linalg.eigvals(mod_pem.A)
                is_stable = np.all(np.abs(evals) < 1.0)
            
            if not is_stable:
                report.append("PEM result unstable. Reverting to initial model.")
            else:
                pem_val_score = mod_pem.compare(y_val, u_val)
                
                if pem_val_score > best_score:
                    final_model = mod_pem
                    final_score = pem_val_score
                    report.append(f"PEM improved Fit: {best_score:.2f}% -> {final_score:.2f}%")
                else:
                    report.append(f"PEM did not improve Validation Fit ({pem_val_score:.2f}%). Reverting.")
                    
        except Exception as e:
            report.append(f"PEM failed ({e}). Keeping tournament winner.")


    # ---------------------------------------------------------
    # 5. FINAL CONVERSION
    # ---------------------------------------------------------
    
    if final_model is not None:
        is_ss = isinstance(final_model, StateSpaceModel)
        
        if result_type == 'polynomial' and is_ss:
            try:
                final_model = final_model.to_transfer_function() 
                report.append("Converted StateSpace -> Polynomial.")
            except:
                report.append("Conversion to Polynomial failed.")
                
        elif result_type == 'state_space' and not is_ss:
            try:
                final_model = final_model.to_state_space()
                report.append("Converted Polynomial -> StateSpace.")
            except:
                report.append("Conversion to StateSpace failed.")


    # ---------------------------------------------------------
    # 6. RETURN
    # ---------------------------------------------------------
    return AutoIdentResult(
        model=final_model,
        data=data,
        metrics={'fit': final_score},
        report=report
    )
