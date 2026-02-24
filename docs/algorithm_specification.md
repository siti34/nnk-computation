# HTO Mechanical Axis Computation - Algorithm Specification

> **Document scope:** This document provides a complete mathematical and algorithmic specification for the HTO (High Tibial Osteotomy) mechanical axis computation pipeline. The pipeline ingests Vicon motion capture marker trajectory data captured on a phantom sawbones knee model and outputs clinically relevant lower-limb alignment angles used in HTO surgical planning.

---

## 1. Goal

Compute the mechanical axis alignment and HTO planning angles of the lower limb from Vicon motion capture marker trajectory data captured on a phantom sawbones knee model. The system uses two rigid marker clusters (femoral: F1--F4, FC; tibial: T1--T4, TC) and a digitizer tool (D1--D4, DT) to register anatomical landmarks and compute clinical angles in 3D.

Specifically, the pipeline must:

1. Establish local coordinate systems (LCS) for each rigid marker cluster from static reference frames.
2. Track the pose (rotation + translation) of each cluster through dynamic frames using the Kabsch algorithm.
3. Transform digitized anatomical landmark positions into global coordinates at every frame.
4. Estimate the hip joint center (HJC) via sphere-fitting of femoral marker trajectories during rotation trials.
5. Compute knee joint center (KJC) and ankle joint center (AJC) from digitized epicondyles and malleoli.
6. Define a frontal (coronal) anatomical plane and project all relevant vectors onto it.
7. Calculate the six standard HTO clinical angles: HKA, mLDFA, MPTA, JLCA, mLDTA, and (where data permits) mLPFA.
8. Quantify measurement uncertainty through rigidity checks, sphere-fit residuals, and Monte Carlo sensitivity analysis.

---

## 2. Experimental Setup

### 2.1 Physical Model

- **Phantom sawbones model:** A commercially available synthetic femur + tibia assembly with anatomically representative geometry. The model eliminates soft tissue artifact, providing a rigid-body ground truth for algorithm validation.

### 2.2 Marker Clusters

| Cluster | Markers | Attachment | Purpose |
|---------|---------|------------|---------|
| Femoral | F1, F2, F3, F4, FC | Rigid plate screwed to the femoral shaft | Track femoral pose |
| Tibial | T1, T2, T3, T4, TC | Rigid plate screwed to the tibial shaft | Track tibial pose |
| Digitizer | D1, D2, D3, D4, DT | Handheld rigid tool | Register anatomical landmarks |

- Each cluster comprises **5 retroreflective markers** on a rigid plate.
- **DT** is the digitizer tip, the point whose position in the digitizer tool's local frame is known and fixed.

### 2.3 Motion Capture System

| Parameter | Value |
|-----------|-------|
| System | Vicon motion capture |
| Sampling frequency | 100 Hz |
| Coordinate units | millimeters (mm) |
| Force plates | AMTI (captured but **not used** in this algorithm) |
| Output format | C3D / CSV marker trajectories |

### 2.4 Trial Protocol

1. **Static trial:** All clusters visible, model stationary. Used to build reference LCS.
2. **Digitization trials:** Digitizer tip placed on each anatomical landmark while femoral and/or tibial clusters remain visible. One trial per landmark.
3. **Rotation trial (HJC):** Femur rotated about the hip joint (ball-and-socket) while the femoral cluster is tracked. Used for sphere-fit HJC estimation.
4. **Dynamic trials (optional):** Full movement trials for dynamic angle computation.

---

## 3. Anatomical Landmarks

Six anatomical landmarks are digitized on the phantom model. Each landmark is registered by placing the digitizer tip (DT) on the bony prominence while the relevant cluster is simultaneously visible.

| # | Abbreviation | Full Name | Anatomical Description | Associated Segment |
|---|-------------|-----------|------------------------|---------------------|
| 1 | **LFEC** | Lateral Femoral Epicondyle | Most prominent point on the lateral femoral epicondyle | Femur |
| 2 | **MFEC** | Medial Femoral Epicondyle | Most prominent point on the medial femoral epicondyle | Femur |
| 3 | **LTP** | Lateral Tibial Plateau | Most lateral point on the proximal articular surface of the tibial plateau | Tibia |
| 4 | **MTP** | Medial Tibial Plateau | Most medial point on the proximal articular surface of the tibial plateau | Tibia |
| 5 | **LM** | Lateral Malleolus | Most prominent point on the lateral malleolus (distal fibula) | Tibia |
| 6 | **MM** | Medial Malleolus | Most prominent point on the medial malleolus (distal tibia) | Tibia |

**Note:** LFEC and MFEC are registered relative to the **femoral** cluster. LTP, MTP, LM, and MM are registered relative to the **tibial** cluster.

---

## 4. Mathematical Formulations

### 4.1 Local Coordinate System (LCS) via SVD

A local coordinate system is established for each rigid marker cluster from static reference frame data. This LCS provides a basis for expressing landmark positions in a body-fixed frame that moves rigidly with the cluster.

#### Input

$N = 5$ markers with positions $\mathbf{p}_i \in \mathbb{R}^3$, $i = 1, \dots, N$, averaged over all static frames to reduce noise.

#### Step 1: Compute the centroid

$$
\mathbf{c} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{p}_i
$$

#### Step 2: Form the centered coordinate matrix

$$
\mathbf{Q} = \begin{bmatrix} (\mathbf{p}_1 - \mathbf{c})^\top \\ (\mathbf{p}_2 - \mathbf{c})^\top \\ \vdots \\ (\mathbf{p}_N - \mathbf{c})^\top \end{bmatrix} \in \mathbb{R}^{N \times 3}
$$

Each row of $\mathbf{Q}$ is the mean-centered position of one marker.

#### Step 3: Singular Value Decomposition

$$
\mathbf{Q} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top
$$

where:
- $\mathbf{U} \in \mathbb{R}^{N \times N}$ contains the left singular vectors,
- $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times 3}$ is the diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \sigma_3 \geq 0$,
- $\mathbf{V} \in \mathbb{R}^{3 \times 3}$ contains the right singular vectors (the principal axes of the marker distribution).

#### Step 4: Construct the reference rotation matrix

$$
\mathbf{R}_{\text{ref}} = \mathbf{V}
$$

The columns of $\mathbf{V}$ form an orthonormal basis aligned with the principal directions of the marker cluster's spatial distribution.

**Right-handedness check:** If $\det(\mathbf{R}_{\text{ref}}) < 0$, negate the third column of $\mathbf{V}$:

$$
\mathbf{R}_{\text{ref}} = \begin{bmatrix} \mathbf{v}_1 & \mathbf{v}_2 & -\mathbf{v}_3 \end{bmatrix}
$$

This ensures $\mathbf{R}_{\text{ref}} \in SO(3)$ (proper rotation).

#### Step 5: Express markers in local coordinates

$$
\mathbf{q}_i^{\text{local}} = \mathbf{R}_{\text{ref}}^\top (\mathbf{p}_i - \mathbf{c})
$$

These local coordinates $\mathbf{q}_i^{\text{local}}$ are invariant under rigid body motion of the cluster and serve as the reference configuration for the Kabsch algorithm.

---

### 4.2 Kabsch Algorithm for Dynamic Tracking

The Kabsch algorithm (Kabsch, 1976) solves for the optimal rotation between two paired point sets in the least-squares sense. It is applied at each dynamic frame to recover the rigid body pose (rotation and translation) of each marker cluster.

#### Input

- Reference local coordinates: $\mathbf{q}_i^{\text{ref}} \in \mathbb{R}^3$, $i = 1, \dots, N$ (from Section 4.1, Step 5)
- Current frame marker positions: $\mathbf{p}_i(t) \in \mathbb{R}^3$, $i = 1, \dots, N$

#### Step 1: Current centroid

$$
\mathbf{c}(t) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{p}_i(t)
$$

#### Step 2: Center current positions

$$
\mathbf{q}_i(t) = \mathbf{p}_i(t) - \mathbf{c}(t)
$$

#### Step 3: Cross-covariance matrix

Construct the matrices:

$$
\mathbf{Q}_{\text{ref}} = \begin{bmatrix} (\mathbf{q}_1^{\text{ref}})^\top \\ \vdots \\ (\mathbf{q}_N^{\text{ref}})^\top \end{bmatrix} \in \mathbb{R}^{N \times 3}, \qquad
\mathbf{Q}(t) = \begin{bmatrix} \mathbf{q}_1(t)^\top \\ \vdots \\ \mathbf{q}_N(t)^\top \end{bmatrix} \in \mathbb{R}^{N \times 3}
$$

The cross-covariance matrix is:

$$
\mathbf{H} = \mathbf{Q}_{\text{ref}}^\top \, \mathbf{Q}(t) \in \mathbb{R}^{3 \times 3}
$$

#### Step 4: SVD of cross-covariance

$$
\mathbf{H} = \mathbf{U}_H \, \mathbf{S}_H \, \mathbf{V}_H^\top
$$

#### Step 5: Optimal rotation

$$
\mathbf{R}(t) = \mathbf{V}_H \, \text{diag}\!\left(1, \; 1, \; \det(\mathbf{V}_H \, \mathbf{U}_H^\top)\right) \, \mathbf{U}_H^\top
$$

The $\text{diag}(1, 1, \det(\mathbf{V}_H \mathbf{U}_H^\top))$ correction ensures $\mathbf{R}(t) \in SO(3)$ (prevents reflections when markers are nearly coplanar).

#### Step 6: Translation

$$
\mathbf{t}(t) = \mathbf{c}(t) - \mathbf{R}(t) \, \mathbf{c}_{\text{ref}}
$$

where $\mathbf{c}_{\text{ref}}$ is the centroid of the reference local coordinates (which is $\mathbf{0}$ if the local coordinates were mean-centered).

#### Step 7: RMSE quality metric

$$
\text{RMSE}(t) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left\| \mathbf{R}(t) \, \mathbf{q}_i^{\text{ref}} - \mathbf{q}_i(t) \right\|^2}
$$

A low RMSE (< 0.5 mm) confirms the rigid body assumption holds. Elevated RMSE indicates marker dropout, mislabeling, or deformation.

#### Step 8: Transform anatomical landmarks to global frame

Given an anatomical landmark with local coordinates $\mathbf{q}^{\text{local}}$ (expressed in the reference LCS), its global position at frame $t$ is:

$$
\mathbf{p}_{\text{global}}(t) = \mathbf{R}(t) \, \mathbf{R}_{\text{ref}} \, \mathbf{q}^{\text{local}} + \mathbf{c}(t)
$$

Equivalently, if we define the full pose rotation as $\mathbf{R}_{\text{pose}}(t) = \mathbf{R}(t) \, \mathbf{R}_{\text{ref}}$:

$$
\mathbf{p}_{\text{global}}(t) = \mathbf{R}_{\text{pose}}(t) \, \mathbf{q}^{\text{local}} + \mathbf{c}(t)
$$

---

### 4.3 Hip Joint Center (HJC) - Pivot / Sphere-Fit Method

The hip joint center is estimated by exploiting the fact that during a rotation trial, each marker on the femoral cluster traces an arc on a sphere centered at the (stationary) hip joint center. This is the functional approach to HJC estimation (Gamage & Lasenby, 2002).

#### Geometric model

Each marker $k$ on the femoral cluster, tracked across $N_f$ frames, has positions $\{\mathbf{p}_k^{(j)}\}_{j=1}^{N_f}$ that lie approximately on a sphere:

$$
\left\| \mathbf{p}_k^{(j)} - \mathbf{c}_{\text{HJC}} \right\| \approx r_k \quad \forall \; j = 1, \dots, N_f
$$

#### 4.3.1 Per-marker sphere fit

For each marker $k$, solve the nonlinear least-squares problem:

$$
\min_{\mathbf{c}_k, \, r_k} \sum_{j=1}^{N_f} \left( \left\| \mathbf{p}_k^{(j)} - \mathbf{c}_k \right\| - r_k \right)^2
$$

**Parameters:** $\mathbf{c}_k = [c_{k,x}, \; c_{k,y}, \; c_{k,z}]^\top$ (sphere center) and $r_k$ (sphere radius).

**Solver:** Levenberg-Marquardt algorithm via `scipy.optimize.least_squares`. The residual vector for the solver is:

$$
\mathbf{r}_k = \begin{bmatrix} \| \mathbf{p}_k^{(1)} - \mathbf{c}_k \| - r_k \\ \vdots \\ \| \mathbf{p}_k^{(N_f)} - \mathbf{c}_k \| - r_k \end{bmatrix} \in \mathbb{R}^{N_f}
$$

**Initial guess:** $\mathbf{c}_k^{(0)} = \text{mean}(\mathbf{p}_k^{(j)})$, $\; r_k^{(0)} = \text{mean}(\| \mathbf{p}_k^{(j)} - \mathbf{c}_k^{(0)} \|)$.

After fitting all markers independently:

$$
\mathbf{c}_{\text{HJC}}^{\text{per-marker}} = \frac{1}{N_{\text{markers}}} \sum_{k=1}^{N_{\text{markers}}} \mathbf{c}_k
$$

#### 4.3.2 Pooled sphere fit

Combine all marker trajectories into a single least-squares problem. For $M$ markers each with $N_f$ frames, the residual vector has $M \times N_f$ elements:

$$
\min_{\mathbf{c}, \, r} \sum_{k=1}^{M} \sum_{j=1}^{N_f} \left( \left\| \mathbf{p}_k^{(j)} - \mathbf{c} \right\| - r \right)^2
$$

**Note:** In the pooled method, a single radius $r$ is fit. This is an approximation since each marker orbits at a different radius. A variant allows per-marker radii $r_k$ while constraining all centers to be equal.

#### 4.3.3 Precision metrics

| Metric | Definition | Acceptable Threshold |
|--------|-----------|---------------------|
| Residual std | $\sigma = \text{std}(\| \mathbf{p}_k^{(j)} - \mathbf{c} \| - r)$ | < 2 mm |
| Center spread | $\text{std}(\mathbf{c}_k)$ across markers | < 5 mm |
| Radius consistency | $\max(r_k) - \min(r_k)$ relative to mean | < 10% of mean radius |

---

### 4.4 Knee Joint Center (KJC)

The knee joint center is defined as the midpoint of the femoral epicondyles:

$$
\mathbf{KJC}(t) = \frac{\mathbf{LFEC}(t) + \mathbf{MFEC}(t)}{2}
$$

where $\mathbf{LFEC}(t)$ and $\mathbf{MFEC}(t)$ are the global positions of the lateral and medial femoral epicondyles at frame $t$, obtained via the Kabsch transform (Section 4.2, Step 8).

---

### 4.5 Ankle Joint Center (AJC)

The ankle joint center is defined as the midpoint of the malleoli:

$$
\mathbf{AJC}(t) = \frac{\mathbf{LM}(t) + \mathbf{MM}(t)}{2}
$$

where $\mathbf{LM}(t)$ and $\mathbf{MM}(t)$ are the global positions of the lateral and medial malleoli at frame $t$.

---

## 5. HTO Clinical Angle Definitions

All clinical angles are computed in the **frontal (coronal) plane** of the lower limb. This section defines the frontal plane construction and each angle in detail.

### 5.1 Frontal Plane Construction

The frontal plane is defined from the femoral mechanical axis and the transepicondylar axis.

#### Step 1: Femoral mechanical axis direction

$$
\mathbf{v}_{\text{mech}} = \mathbf{HJC} - \mathbf{KJC}
$$

This vector points from the knee toward the hip (distal to proximal).

#### Step 2: Medial-lateral direction (transepicondylar axis)

$$
\mathbf{v}_{\text{ml}} = \mathbf{LFEC} - \mathbf{MFEC}
$$

This vector points from medial to lateral.

#### Step 3: Frontal plane normal

$$
\mathbf{n}_{\text{frontal}} = \frac{\mathbf{v}_{\text{mech}} \times \mathbf{v}_{\text{ml}}}{\| \mathbf{v}_{\text{mech}} \times \mathbf{v}_{\text{ml}} \|}
$$

The normal $\mathbf{n}_{\text{frontal}}$ points **anteriorly** (by the right-hand rule, given the convention that $\mathbf{v}_{\text{mech}}$ points superiorly and $\mathbf{v}_{\text{ml}}$ points laterally for a right leg).

#### Projection operator

To project any vector $\mathbf{v}$ onto the frontal plane:

$$
\mathbf{v}_{\text{proj}} = \mathbf{v} - (\mathbf{v} \cdot \mathbf{n}_{\text{frontal}}) \, \mathbf{n}_{\text{frontal}}
$$

#### Angle between two projected vectors

Given two vectors $\mathbf{a}$ and $\mathbf{b}$ projected onto the frontal plane, the signed angle (measured in the frontal plane) is:

$$
\theta = \text{atan2}\!\left((\mathbf{a}_{\text{proj}} \times \mathbf{b}_{\text{proj}}) \cdot \mathbf{n}_{\text{frontal}}, \; \mathbf{a}_{\text{proj}} \cdot \mathbf{b}_{\text{proj}}\right)
$$

The unsigned angle between the two vectors is:

$$
\theta = \arccos\!\left(\frac{\mathbf{a}_{\text{proj}} \cdot \mathbf{b}_{\text{proj}}}{\| \mathbf{a}_{\text{proj}} \| \; \| \mathbf{b}_{\text{proj}} \|}\right)
$$

---

### 5.2 HKA (Hip-Knee-Ankle Angle)

The HKA angle is the primary metric for overall lower-limb alignment. It is the angle at the knee between the femoral and tibial mechanical axes, measured in the frontal plane.

#### Definition

- **Femoral mechanical axis:** $\mathbf{v}_{\text{femur}} = \mathbf{HJC} - \mathbf{KJC}$ (points proximally from knee to hip)
- **Tibial mechanical axis:** $\mathbf{v}_{\text{tibia}} = \mathbf{AJC} - \mathbf{KJC}$ (points distally from knee to ankle)

#### Computation

1. Project both vectors onto the frontal plane:

$$
\mathbf{v}_{\text{femur}}^{\text{proj}} = \mathbf{v}_{\text{femur}} - (\mathbf{v}_{\text{femur}} \cdot \mathbf{n}_{\text{frontal}}) \, \mathbf{n}_{\text{frontal}}
$$

$$
\mathbf{v}_{\text{tibia}}^{\text{proj}} = \mathbf{v}_{\text{tibia}} - (\mathbf{v}_{\text{tibia}} \cdot \mathbf{n}_{\text{frontal}}) \, \mathbf{n}_{\text{frontal}}
$$

2. Compute the angle at the knee vertex:

$$
\text{HKA} = \arccos\!\left(\frac{\mathbf{v}_{\text{femur}}^{\text{proj}} \cdot \mathbf{v}_{\text{tibia}}^{\text{proj}}}{\| \mathbf{v}_{\text{femur}}^{\text{proj}} \| \; \| \mathbf{v}_{\text{tibia}}^{\text{proj}} \|}\right)
$$

This gives the supplement of the conventional HKA. The full HKA as a straight-line angle is:

$$
\text{HKA}_{\text{full}} = 180^\circ - \text{HKA} \quad \text{(if using the acute angle between the two axes)}
$$

Or equivalently, using the signed angle convention:

$$
\text{HKA} = 180^\circ \quad \text{for neutral (collinear) alignment}
$$

#### Clinical interpretation

| HKA Value | Alignment | Description |
|-----------|-----------|-------------|
| $= 180°$ | Neutral | Mechanical axes are collinear |
| $< 180°$ | **Varus** | Tibia angled medially (bowlegged) |
| $> 180°$ | **Valgus** | Tibia angled laterally (knock-kneed) |
| $180° \pm 3°$ | Normal range | Clinically acceptable alignment |

---

### 5.3 mLDFA (mechanical Lateral Distal Femoral Angle)

The mLDFA quantifies the orientation of the distal femoral joint line relative to the femoral mechanical axis, measured on the **lateral** side.

#### Definition

- **Femoral mechanical axis:** $\mathbf{v}_{\text{fem}} = \mathbf{HJC} \to \mathbf{KJC}$ (proximal to distal direction)

$$
\mathbf{v}_{\text{fem}} = \mathbf{KJC} - \mathbf{HJC}
$$

- **Distal femoral joint line:** $\mathbf{v}_{\text{dfj}} = \mathbf{MFEC} \to \mathbf{LFEC}$ (medial to lateral direction)

$$
\mathbf{v}_{\text{dfj}} = \mathbf{LFEC} - \mathbf{MFEC}
$$

#### Computation

1. Project both vectors onto the frontal plane.
2. Compute the lateral angle between them:

$$
\text{mLDFA} = \arccos\!\left(\frac{\mathbf{v}_{\text{fem}}^{\text{proj}} \cdot \mathbf{v}_{\text{dfj}}^{\text{proj}}}{\| \mathbf{v}_{\text{fem}}^{\text{proj}} \| \; \| \mathbf{v}_{\text{dfj}}^{\text{proj}} \|}\right)
$$

The angle is measured between the mechanical axis (directed distally) and the joint line (directed laterally), on the **lateral** side of the mechanical axis.

#### Normal values

$$
\text{mLDFA}_{\text{normal}} = 85°\text{--}90° \quad (\text{typically } 87°)
$$

---

### 5.4 MPTA (Medial Proximal Tibial Angle)

The MPTA quantifies the orientation of the proximal tibial joint line relative to the tibial mechanical axis, measured on the **medial** side. This is a critical angle for HTO planning.

#### Definition

- **Tibial mechanical axis:** $\mathbf{v}_{\text{tib}} = \mathbf{KJC} \to \mathbf{AJC}$ (proximal to distal direction)

$$
\mathbf{v}_{\text{tib}} = \mathbf{AJC} - \mathbf{KJC}
$$

- **Proximal tibial joint line:** $\mathbf{v}_{\text{ptj}} = \mathbf{LTP} \to \mathbf{MTP}$ (lateral to medial direction)

$$
\mathbf{v}_{\text{ptj}} = \mathbf{MTP} - \mathbf{LTP}
$$

#### Computation

1. Project both vectors onto the frontal plane.
2. Compute the medial angle between them:

$$
\text{MPTA} = \arccos\!\left(\frac{\mathbf{v}_{\text{tib}}^{\text{proj}} \cdot \mathbf{v}_{\text{ptj}}^{\text{proj}}}{\| \mathbf{v}_{\text{tib}}^{\text{proj}} \| \; \| \mathbf{v}_{\text{ptj}}^{\text{proj}} \|}\right)
$$

The angle is measured between the tibial mechanical axis (directed distally) and the proximal joint line (directed medially), on the **medial** side of the mechanical axis.

#### Normal values

$$
\text{MPTA}_{\text{normal}} = 85°\text{--}90° \quad (\text{typically } 87°)
$$

#### HTO planning significance

In medial opening-wedge HTO, the correction angle is directly derived from the MPTA deficit:

$$
\text{Correction} \approx \text{MPTA}_{\text{target}} - \text{MPTA}_{\text{measured}}
$$

---

### 5.5 JLCA (Joint Line Convergence Angle)

The JLCA measures the angular convergence between the distal femoral and proximal tibial joint lines. It reflects the contribution of soft tissue laxity (or cartilage/bone loss) to the overall deformity.

#### Definition

- **Distal femoral joint line:** $\mathbf{v}_{\text{dfj}} = \mathbf{MFEC} \to \mathbf{LFEC}$

$$
\mathbf{v}_{\text{dfj}} = \mathbf{LFEC} - \mathbf{MFEC}
$$

- **Proximal tibial joint line:** $\mathbf{v}_{\text{ptj}} = \mathbf{MTP} \to \mathbf{LTP}$

$$
\mathbf{v}_{\text{ptj}} = \mathbf{LTP} - \mathbf{MTP}
$$

#### Computation

1. Project both vectors onto the frontal plane.
2. Compute the angle between the two joint lines:

$$
\text{JLCA} = \arccos\!\left(\frac{\mathbf{v}_{\text{dfj}}^{\text{proj}} \cdot \mathbf{v}_{\text{ptj}}^{\text{proj}}}{\| \mathbf{v}_{\text{dfj}}^{\text{proj}} \| \; \| \mathbf{v}_{\text{ptj}}^{\text{proj}} \|}\right)
$$

A sign convention is applied: positive JLCA indicates convergence on the medial side (consistent with varus); negative indicates lateral convergence (valgus).

#### Normal values

$$
\text{JLCA}_{\text{normal}} = 0°\text{--}2°
$$

**Note:** On a rigid phantom model with no cartilage or ligamentous laxity, the JLCA should be very close to $0°$.

---

### 5.6 mLDTA (mechanical Lateral Distal Tibial Angle)

The mLDTA quantifies the orientation of the distal tibial joint line (ankle joint line) relative to the tibial mechanical axis, measured on the **lateral** side.

#### Definition

- **Tibial mechanical axis:** $\mathbf{v}_{\text{tib}} = \mathbf{KJC} \to \mathbf{AJC}$ (proximal to distal direction)

$$
\mathbf{v}_{\text{tib}} = \mathbf{AJC} - \mathbf{KJC}
$$

- **Distal tibial joint line:** $\mathbf{v}_{\text{dtj}} = \mathbf{MM} \to \mathbf{LM}$ (medial to lateral direction)

$$
\mathbf{v}_{\text{dtj}} = \mathbf{LM} - \mathbf{MM}
$$

#### Computation

1. Project both vectors onto the frontal plane.
2. Compute the lateral angle:

$$
\text{mLDTA} = \arccos\!\left(\frac{\mathbf{v}_{\text{tib}}^{\text{proj}} \cdot \mathbf{v}_{\text{dtj}}^{\text{proj}}}{\| \mathbf{v}_{\text{tib}}^{\text{proj}} \| \; \| \mathbf{v}_{\text{dtj}}^{\text{proj}} \|}\right)
$$

#### Normal values

$$
\text{mLDTA}_{\text{normal}} \approx 89°
$$

---

### 5.7 mLPFA (mechanical Lateral Proximal Femoral Angle)

The mLPFA is the angle between the femoral mechanical axis and the proximal femoral joint orientation line, measured on the lateral side.

**NOT COMPUTABLE** with the current marker set. The mLPFA requires the **greater trochanter** landmark, which is not digitized in this protocol. The greater trochanter is needed to define the proximal femoral joint orientation line (from the hip joint center to the greater trochanter tip).

#### Normal values (for reference)

$$
\text{mLPFA}_{\text{normal}} = 90° \pm 5°
$$

#### Note

If the greater trochanter is digitized in a future protocol revision, the mLPFA can be computed as:

- Proximal femoral joint line: $\mathbf{v}_{\text{pfj}} = \mathbf{HJC} \to \mathbf{GT}$ (hip center to greater trochanter)
- Femoral mechanical axis: $\mathbf{v}_{\text{fem}} = \mathbf{HJC} \to \mathbf{KJC}$
- mLPFA = lateral angle between these two vectors, projected onto the frontal plane

---

### 5.8 Algebraic Relationship Between Angles

The HKA angle is algebraically related to the other frontal plane angles through the following identity (Paley, 2002):

$$
\boxed{\text{HKA deviation} = \text{mLDFA} - \text{MPTA} + \text{JLCA}}
$$

where HKA deviation is defined as $180° - \text{HKA}$ (positive for varus, negative for valgus).

Equivalently:

$$
180° - \text{HKA} \approx \text{mLDFA} - \text{MPTA} + \text{JLCA}
$$

**Derivation sketch:** The total angular deviation of the mechanical axis at the knee is composed of:
1. The femoral contribution ($90° - \text{mLDFA}$): how much the femoral mechanical axis deviates from perpendicular to the joint line,
2. The tibial contribution ($90° - \text{MPTA}$): how much the tibial mechanical axis deviates from perpendicular to the joint line,
3. The joint line convergence ($\text{JLCA}$): the angular mismatch between the femoral and tibial joint lines themselves.

This relationship serves as an **internal consistency check**: if the computed angles do not approximately satisfy this identity, it indicates a measurement or computational error.

---

## 6. Error Analysis

### 6.1 Sources of Error

| # | Error Source | Typical Magnitude | Notes |
|---|-------------|-------------------|-------|
| 1 | **Marker reconstruction error** | < 0.5 mm | Vicon system noise; depends on camera count, volume, and calibration quality |
| 2 | **Rigid body assumption violation** | Negligible (phantom) | Soft tissue artifact is the dominant error in vivo; not applicable for phantom sawbones |
| 3 | **Digitizer tip placement accuracy** | 1--3 mm | Depends on operator skill, landmark prominence, and tip geometry |
| 4 | **Sphere-fit accuracy (HJC)** | 1--5 mm | Depends on arc range, number of frames, and motion quality |
| 5 | **Frontal plane definition sensitivity** | 0.5--2° | The frontal plane depends on the transepicondylar axis which has ~2--3 mm uncertainty |

---

### 6.2 Rigidity Validation

The rigid body assumption is verified by monitoring inter-marker distances throughout each trial.

#### Method

For each pair of markers $(i, j)$ in a cluster, compute the Euclidean distance at each frame:

$$
d_{ij}(t) = \| \mathbf{p}_i(t) - \mathbf{p}_j(t) \|
$$

Compute the reference distance from the static trial:

$$
d_{ij}^{\text{ref}} = \text{mean}_t \left( d_{ij}(t) \right) \quad \text{(over static frames)}
$$

The RMS deviation for each pair is:

$$
\Delta d_{ij} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \left( d_{ij}(t) - d_{ij}^{\text{ref}} \right)^2}
$$

The overall rigidity metric for a cluster is:

$$
\Delta d_{\text{RMS}} = \sqrt{\frac{1}{N_{\text{pairs}}} \sum_{(i,j)} \Delta d_{ij}^2}
$$

where $N_{\text{pairs}} = \binom{N}{2} = 10$ for a 5-marker cluster.

#### Acceptance criteria

| Metric | Threshold | Action if exceeded |
|--------|-----------|-------------------|
| $\Delta d_{\text{RMS}}$ per cluster | < 0.5 mm | Acceptable; rigid body assumption holds |
| $\Delta d_{ij}$ for any pair | > 1.0 mm | Flag specific marker pair for investigation |
| Individual frame $|d_{ij}(t) - d_{ij}^{\text{ref}}|$ | > 2.0 mm | Flag frame; possible marker dropout or mislabeling |

---

### 6.3 HJC Precision

#### Pooled sphere-fit residuals

After fitting the sphere to the pooled marker data, compute the residual for each observation:

$$
e_k^{(j)} = \left\| \mathbf{p}_k^{(j)} - \mathbf{c}_{\text{HJC}} \right\| - r
$$

The standard deviation of these residuals should be:

$$
\sigma_{\text{residual}} = \text{std}(e_k^{(j)}) < 2 \; \text{mm}
$$

#### Per-marker center spread

Fit individual spheres to each marker's trajectory and compare the resulting centers:

$$
\text{Center spread} = \text{std}\left(\{\mathbf{c}_k\}_{k=1}^{M}\right) < 5 \; \text{mm} \quad \text{(in each axis)}
$$

#### Radius consistency

The radii from per-marker fits should be consistent:

$$
\frac{\max(r_k) - \min(r_k)}{\text{mean}(r_k)} < 0.10 \quad \text{(< 10\%)}
$$

Large discrepancies suggest insufficient arc range, non-spherical motion, or marker mislabeling.

---

### 6.4 Landmark Digitization Repeatability

If the same anatomical landmark is digitized in multiple trials (repeated digitizations), the spatial spread quantifies digitization precision.

#### Method

Given $K$ repeated digitizations of the same landmark, each yielding position $\mathbf{l}_k$:

$$
\bar{\mathbf{l}} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{l}_k
$$

$$
\text{Repeatability} = \sqrt{\frac{1}{K} \sum_{k=1}^{K} \| \mathbf{l}_k - \bar{\mathbf{l}} \|^2}
$$

#### Acceptance criterion

$$
\text{Repeatability} < 2 \; \text{mm}
$$

This threshold is appropriate for well-defined bony prominences on a phantom model. In vivo, the threshold is typically relaxed to 3--5 mm due to soft tissue coverage.

---

### 6.5 Angle Sensitivity Analysis (Monte Carlo)

A Monte Carlo simulation is used to propagate landmark position uncertainty through the angle computations and quantify the resulting angle uncertainty.

#### Method

1. For each of the 6 anatomical landmarks plus HJC (7 points total), define the position uncertainty as isotropic Gaussian noise:

$$
\mathbf{l}_{\text{perturbed}} = \mathbf{l}_{\text{nominal}} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_3)
$$

where $\sigma = 0.5$ mm (corresponding to approximately $\pm 1$ mm at the 95% confidence level).

2. For each of $N_{\text{MC}} = 1000$ iterations:
   a. Perturb all landmark positions independently.
   b. Recompute KJC, AJC, and all clinical angles.
   c. Store the resulting angle values.

3. Report statistics for each angle $\alpha$:

$$
\bar{\alpha} = \frac{1}{N_{\text{MC}}} \sum_{i=1}^{N_{\text{MC}}} \alpha_i
$$

$$
\sigma_\alpha = \sqrt{\frac{1}{N_{\text{MC}} - 1} \sum_{i=1}^{N_{\text{MC}}} (\alpha_i - \bar{\alpha})^2}
$$

$$
\text{95\% CI} = \left[\text{percentile}_{2.5}(\alpha_i), \; \text{percentile}_{97.5}(\alpha_i)\right]
$$

#### Expected sensitivity

| Angle | Typical $\sigma_\alpha$ per mm of landmark perturbation |
|-------|--------------------------------------------------------|
| HKA | 0.5--1.0° |
| mLDFA | 0.3--0.8° |
| MPTA | 0.3--0.8° |
| JLCA | 0.2--0.5° |
| mLDTA | 0.3--0.8° |

The HKA angle is most sensitive because it depends on HJC (which has larger uncertainty due to the sphere-fit estimation) and the longest lever arms.

---

### 6.6 Dynamic Stability

For quasi-static portions of trials (where the phantom model is stationary or moving slowly), the computed angles should remain nearly constant.

#### Method

1. Identify quasi-static windows (e.g., first and last 50 frames of a trial, or frames where marker velocity < 1 mm/frame).
2. Compute each clinical angle at every frame within the window.
3. Report the standard deviation:

$$
\sigma_{\alpha}^{\text{dynamic}} = \text{std}\left(\{\alpha(t)\}_{t \in \text{static window}}\right)
$$

#### Acceptance criterion

$$
\sigma_{\alpha}^{\text{dynamic}} < 1° \quad \text{for all angles}
$$

For a rigid phantom model with well-tracked markers, values of $\sigma_{\alpha}^{\text{dynamic}} < 0.5°$ are expected. Values exceeding $1°$ suggest marker tracking issues or incorrect landmark registration.

---

## 7. Evaluation Method

The accuracy and reliability of the pipeline are assessed through the following five evaluation strategies.

### 7.1 Intra-trial Repeatability

Compute the standard deviation of each clinical angle across static frames within a single trial:

$$
\sigma_{\text{intra}} = \text{std}\left(\{\alpha(t)\}_{t=1}^{T}\right)
$$

**Expected:** $\sigma_{\text{intra}} < 0.5°$ for a rigid phantom with good marker visibility.

### 7.2 Inter-trial Reproducibility

Compare the mean angle values across multiple dynamic trials with the same setup:

$$
\bar{\alpha}_k = \text{mean}_{t}(\alpha_k(t)) \quad \text{for trial } k
$$

$$
\sigma_{\text{inter}} = \text{std}\left(\{\bar{\alpha}_k\}_{k=1}^{K}\right)
$$

**Expected:** $\sigma_{\text{inter}} < 1°$ for repeated trials without re-digitization.

### 7.3 Known Geometry Validation

If the phantom bone angles are known from manufacturer specifications or independent measurement (e.g., CT scan, goniometer):

$$
\text{Accuracy} = |\alpha_{\text{computed}} - \alpha_{\text{known}}|
$$

**Target:** Accuracy within $\pm 2°$ of the known geometry.

### 7.4 Kabsch RMSE as Quality Metric

The Kabsch RMSE (Section 4.2, Step 7) serves as a per-frame quality gate:

| RMSE Range | Interpretation |
|------------|---------------|
| < 0.3 mm | Excellent tracking |
| 0.3--0.5 mm | Good tracking |
| 0.5--1.0 mm | Acceptable; inspect markers |
| > 1.0 mm | Poor; likely marker dropout or mislabeling |

Frames with RMSE > 1.0 mm should be excluded from angle computations or flagged for manual review.

### 7.5 HJC Precision from Sphere-fit Residuals

As detailed in Section 6.3, the sphere-fit residual standard deviation and per-marker center spread provide direct estimates of HJC estimation quality. These metrics should be reported alongside the computed angles.

---

## 8. References

1. **Paley D.** *Principles of Deformity Correction.* Springer-Verlag, Berlin, 2002. ISBN 3-540-41665-X.
   - Defines the standard lower-limb alignment angles (mLDFA, MPTA, JLCA, mLDTA, mLPFA) and the algebraic relationship HKA deviation = mLDFA - MPTA + JLCA.

2. **Grood ES, Suntay WJ.** A joint coordinate system for the clinical description of three-dimensional motions: application to the knee. *Journal of Biomechanical Engineering.* 1983;105(2):136--144. doi:10.1115/1.3138397.
   - Establishes the joint coordinate system framework for describing 3D joint kinematics in clinical terms.

3. **Soderkvist I, Wedin PA.** Determining the movements of the skeleton using well-configured markers. *Journal of Biomechanics.* 1993;26(12):1473--1477. doi:10.1016/0021-9290(93)90098-Y.
   - Describes the SVD-based method for determining rigid body motion from marker data, forming the basis for the local coordinate system construction and Kabsch tracking approach.

4. **Gamage SSHU, Lasenby J.** New least squares solutions for estimating the average centre of rotation and the axis of rotation. *Journal of Biomechanics.* 2002;35(1):87--93. doi:10.1016/S0021-9290(01)00160-2.
   - Proposes functional methods for estimating joint centers (including the hip) from marker trajectories during rotational movements.

5. **Kabsch W.** A solution for the best rotation to relate two sets of vectors. *Acta Crystallographica Section A.* 1976;32(5):922--923. doi:10.1107/S0567739476001873.
   - Original paper describing the SVD-based algorithm for finding the optimal rotation matrix between two corresponding point sets in the least-squares sense.

6. **Ehrig RM, Taylor WR, Duda GN, Heller MO.** A survey of formal methods for determining the centre of rotation of ball joints. *Journal of Biomechanics.* 2006;39(15):2798--2809. doi:10.1016/j.jbiomech.2005.10.002.
   - Comprehensive comparison of functional hip joint center estimation methods including sphere-fitting and geometric approaches.

7. **Miniaci A, Ballmer FT, Ballmer PM, Jakob RP.** Proximal tibial osteotomy: a new fixation device. *Clinical Orthopaedics and Related Research.* 1989;(246):250--259.
   - Clinical context for HTO planning and the importance of accurate mechanical axis determination.

---

*Document generated for the HTO Mechanical Axis Computation project. All mathematical formulations correspond to the implementation in the project source code.*
