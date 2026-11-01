"""
Traffic Sign Recognition System - Configuration & Metadata
Defines classes, categories, driving instructions, meanings, and UI styles.
"""

from typing import Dict, Any

# Primary Categories
CATEGORY_SPEED_LIMIT = "Speed Limit"
CATEGORY_STOP_YIELD = "Stop & Yield"
CATEGORY_PROHIBITORY = "No Entry & Prohibitory"
CATEGORY_MANDATORY = "Turn & Mandatory"
CATEGORY_WARNING = "Danger & Warning"

# Color mappings for categories (HEX & RGB for OpenCV)
CATEGORY_COLORS: Dict[str, Dict[str, Any]] = {
    CATEGORY_SPEED_LIMIT: {
        "hex": "#FF3366",
        "bgr": (102, 51, 255),
        "badge": "badge-speed",
        "icon": "⚡"
    },
    CATEGORY_STOP_YIELD: {
        "hex": "#EF4444",
        "bgr": (68, 68, 239),
        "badge": "badge-stop",
        "icon": "🛑"
    },
    CATEGORY_PROHIBITORY: {
        "hex": "#F97316",
        "bgr": (22, 115, 249),
        "badge": "badge-prohibit",
        "icon": "⛔"
    },
    CATEGORY_MANDATORY: {
        "hex": "#3B82F6",
        "bgr": (246, 130, 59),
        "badge": "badge-turn",
        "icon": "🔄"
    },
    CATEGORY_WARNING: {
        "hex": "#F59E0B",
        "bgr": (11, 158, 245),
        "badge": "badge-warn",
        "icon": "⚠️"
    }
}

# 43 Standard GTSRB Classes with comprehensive driving guidelines & meanings
TRAFFIC_SIGN_CLASSES: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "Speed Limit (20 km/h)",
        "category": CATEGORY_SPEED_LIMIT,
        "meaning": "Maximum legally permitted vehicle speed is 20 kilometers per hour.",
        "instruction": "Shift into low gear, maintain extreme caution, and yield to pedestrians in living zones.",
        "penalty_risk": "High if speeding in designated calm/school zones.",
        "shape": "Circular with red border, white background",
        "priority": "Crucial Safety Zone"
    },
    1: {
        "name": "Speed Limit (30 km/h)",
        "category": CATEGORY_SPEED_LIMIT,
        "meaning": "Maximum legally permitted vehicle speed is 30 kilometers per hour.",
        "instruction": "Slow down to 30 km/h. Common in residential and school areas with frequent pedestrian crossings.",
        "penalty_risk": "Moderate to High near active school hours.",
        "shape": "Circular with red border, white background",
        "priority": "Residential Caution"
    },
    2: {
        "name": "Speed Limit (50 km/h)",
        "category": CATEGORY_SPEED_LIMIT,
        "meaning": "Standard urban area maximum speed limit is 50 kilometers per hour.",
        "instruction": "Drive smoothly at or below 50 km/h. Watch for sudden braking, cross traffic, and cyclists.",
        "penalty_risk": "Standard traffic infraction fine.",
        "shape": "Circular with red border, white background",
        "priority": "Urban Standard"
    },
    3: {
        "name": "Speed Limit (60 km/h)",
        "category": CATEGORY_SPEED_LIMIT,
        "meaning": "Maximum permitted speed is 60 kilometers per hour on designated arterial roads.",
        "instruction": "Maintain vehicle speed up to 60 km/h. Keep safe following distance of at least 3 seconds.",
        "penalty_risk": "Standard municipal speeding fine.",
        "shape": "Circular with red border, white background",
        "priority": "Arterial Road"
    },
    4: {
        "name": "Speed Limit (70 km/h)",
        "category": CATEGORY_SPEED_LIMIT,
        "meaning": "Maximum permitted speed is 70 kilometers per hour on suburban connector routes.",
        "instruction": "Maintain steady cruising speed under 70 km/h. Prepare for occasional traffic signals or merges.",
        "penalty_risk": "Fine and license points.",
        "shape": "Circular with red border, white background",
        "priority": "Suburban Transit"
    },
    5: {
        "name": "Speed Limit (80 km/h)",
        "category": CATEGORY_SPEED_LIMIT,
        "meaning": "Maximum permitted speed is 80 kilometers per hour on rural or ring roads.",
        "instruction": "Cruising limit is 80 km/h. Increase following distance and monitor oncoming overtaking traffic.",
        "penalty_risk": "Fine and license penalty points.",
        "shape": "Circular with red border, white background",
        "priority": "Rural Highway"
    },
    6: {
        "name": "End of Speed Limit (80 km/h)",
        "category": CATEGORY_SPEED_LIMIT,
        "meaning": "The previous 80 km/h speed restriction has ended.",
        "instruction": "Default territorial speed limit applies. Accelerate smoothly according to road conditions.",
        "penalty_risk": "Low unless default limit is exceeded.",
        "shape": "Circular with grey diagonal slash",
        "priority": "Deregulation"
    },
    7: {
        "name": "Speed Limit (100 km/h)",
        "category": CATEGORY_SPEED_LIMIT,
        "meaning": "Maximum permitted speed is 100 kilometers per hour on expressways.",
        "instruction": "High-speed highway zone. Stay in right lanes unless overtaking; check blind spots frequently.",
        "penalty_risk": "High fine and severe point penalty if exceeded.",
        "shape": "Circular with red border, white background",
        "priority": "Expressway"
    },
    8: {
        "name": "Speed Limit (120 km/h)",
        "category": CATEGORY_SPEED_LIMIT,
        "meaning": "Maximum permitted speed is 120 kilometers per hour on motorways / autobahns.",
        "instruction": "Maximum motorway speed. Maintain at least 60-meter gap from preceding vehicles. Never stop on shoulder.",
        "penalty_risk": "Severe legal penalties for reckless excess.",
        "shape": "Circular with red border, white background",
        "priority": "Motorway Peak"
    },
    9: {
        "name": "No Passing / Overtaking",
        "category": CATEGORY_PROHIBITORY,
        "meaning": "Overtaking motor vehicles other than two-wheeled mopeds is strictly prohibited.",
        "instruction": "Remain in your current travel lane behind the vehicle ahead. Do not cross centerline to overtake.",
        "penalty_risk": "Heavy fine, mandatory points, and potential license suspension.",
        "shape": "Circular with red border, red and black cars",
        "priority": "Critical Safety Rule"
    },
    10: {
        "name": "No Passing for Heavy Vehicles",
        "category": CATEGORY_PROHIBITORY,
        "meaning": "Goods vehicles exceeding 3.5 tonnes maximum gross weight may not overtake.",
        "instruction": "Trucks and heavy transport must stay in right lane. Passenger cars may proceed normally.",
        "penalty_risk": "Commercial carrier violation fine.",
        "shape": "Circular with red border, red truck and black car",
        "priority": "Freight Regulation"
    },
    11: {
        "name": "Right-of-Way at Next Intersection",
        "category": CATEGORY_STOP_YIELD,
        "meaning": "You hold right-of-way priority at the upcoming side-street intersection.",
        "instruction": "You may proceed through intersection, but stay alert for drivers failing to yield from cross streets.",
        "penalty_risk": "Low, but defensive driving required.",
        "shape": "Triangular warning sign with thick cross-arrow",
        "priority": "Intersection Priority"
    },
    12: {
        "name": "Priority Road",
        "category": CATEGORY_STOP_YIELD,
        "meaning": "You are traveling on a priority road. Cross-street traffic must yield to you.",
        "instruction": "Maintain steady progress. You have the right of way until the End of Priority Road sign appears.",
        "penalty_risk": "Low.",
        "shape": "Yellow diamond with white outer border",
        "priority": "Primary Arterial"
    },
    13: {
        "name": "Yield / Give Way",
        "category": CATEGORY_STOP_YIELD,
        "meaning": "You must give way and yield right-of-way to vehicles on the intersecting road.",
        "instruction": "Slow down, scan both directions, and come to a complete stop if cross traffic or pedestrians are approaching.",
        "penalty_risk": "High penalty for failure to give right-of-way causing hazard.",
        "shape": "Inverted triangle, red border, white background",
        "priority": "Mandatory Yield"
    },
    14: {
        "name": "Stop",
        "category": CATEGORY_STOP_YIELD,
        "meaning": "Complete stop mandatory before the stop line or intersection entrance.",
        "instruction": "Bring wheels to a total standstill for at least 3 seconds. Check left, right, and left again before proceeding.",
        "penalty_risk": "Severe fine and immediate points for rolling stop.",
        "shape": "Red Octagon with white STOP lettering",
        "priority": "Absolute Stop"
    },
    15: {
        "name": "No Vehicles / Road Closed",
        "category": CATEGORY_PROHIBITORY,
        "meaning": "All vehicular traffic is prohibited in both directions.",
        "instruction": "Do not enter. Road is reserved for pedestrians or closed for public events/repairs.",
        "penalty_risk": "Substantial municipal infraction fine.",
        "shape": "Circular with thick red border and blank white center",
        "priority": "Total Restriction"
    },
    16: {
        "name": "Heavy Vehicles Prohibited (>3.5t)",
        "category": CATEGORY_PROHIBITORY,
        "meaning": "Vehicles exceeding 3.5 metric tons gross mass cannot enter this roadway.",
        "instruction": "Truck drivers must reroute via designated bypass. Standard cars may enter.",
        "penalty_risk": "Heavy vehicle route violation.",
        "shape": "Circular with red border and truck silhouette",
        "priority": "Weight Restriction"
    },
    17: {
        "name": "No Entry",
        "category": CATEGORY_PROHIBITORY,
        "meaning": "Do not enter! Traffic moves exclusively in the opposite direction (one-way street).",
        "instruction": "Under no circumstances enter this street. Immediate risk of head-on collision.",
        "penalty_risk": "Severe fine, immediate points, potential license revocation.",
        "shape": "Red circle with horizontal white bar",
        "priority": "Extreme Danger"
    },
    18: {
        "name": "General Caution / Danger Ahead",
        "category": CATEGORY_WARNING,
        "meaning": "Unspecified hazard or dangerous road conditions ahead.",
        "instruction": "Reduce speed, pay full attention to the road, and anticipate unexpected hazards.",
        "penalty_risk": "Moderate if negligence causes incident.",
        "shape": "Triangular with red border and exclamation mark",
        "priority": "Heightened Vigilance"
    },
    19: {
        "name": "Dangerous Curve to the Left",
        "category": CATEGORY_WARNING,
        "meaning": "Sharp or dangerous curve bending toward the left ahead.",
        "instruction": "Brake before entering the turn. Do not brake sharply mid-curve; accelerate gently upon exiting.",
        "penalty_risk": "Loss of control risk.",
        "shape": "Triangular with red border, left curve arrow",
        "priority": "Curve Advisory"
    },
    20: {
        "name": "Dangerous Curve to the Right",
        "category": CATEGORY_WARNING,
        "meaning": "Sharp or dangerous curve bending toward the right ahead.",
        "instruction": "Slow down to safe cornering speed. Stay centered in your lane and watch for road crests.",
        "penalty_risk": "Rollover / skid hazard.",
        "shape": "Triangular with red border, right curve arrow",
        "priority": "Curve Advisory"
    },
    21: {
        "name": "Double Curve (First to Left)",
        "category": CATEGORY_WARNING,
        "meaning": "Series of successive sharp bends ahead, beginning with a turn to the left.",
        "instruction": "Moderate speed significantly. Avoid overtaking through winding multi-curve sections.",
        "penalty_risk": "Skid risk.",
        "shape": "Triangular with red border, winding s-curve symbol",
        "priority": "Winding Road"
    },
    22: {
        "name": "Bumpy / Rough Road",
        "category": CATEGORY_WARNING,
        "meaning": "Uneven pavement surface, potholes, or speed bumps ahead.",
        "instruction": "Reduce speed to protect vehicle suspension and prevent losing tire traction or bottoming out.",
        "penalty_risk": "Vehicle damage / loss of steering control.",
        "shape": "Triangular with red border and bump silhouettes",
        "priority": "Road Quality Warning"
    },
    23: {
        "name": "Slippery Road",
        "category": CATEGORY_WARNING,
        "meaning": "Road surface is prone to become exceptionally slippery in wet, icy, or dirty conditions.",
        "instruction": "Double your braking distance. Avoid abrupt steering inputs or heavy braking.",
        "penalty_risk": "Severe hydroplaning / collision risk.",
        "shape": "Triangular with skidding car graphic",
        "priority": "Traction Alert"
    },
    24: {
        "name": "Road Narrows on Right",
        "category": CATEGORY_WARNING,
        "meaning": "The carriageway narrows ahead on the right side.",
        "instruction": "Merge safely into remaining lane using zipper method. Yield gracefully to merging vehicles.",
        "penalty_risk": "Sideswipe collision risk.",
        "shape": "Triangular with tapering right border line",
        "priority": "Lane Merge"
    },
    25: {
        "name": "Road Work Ahead",
        "category": CATEGORY_WARNING,
        "meaning": "Construction, repair crews, or machinery operating on or alongside the roadway.",
        "instruction": "Slow down immediately, follow construction worker signals, and expect narrow detour lanes.",
        "penalty_risk": "Double fines in active construction zones.",
        "shape": "Triangular with digging worker silhouette",
        "priority": "Work Zone"
    },
    26: {
        "name": "Traffic Signals Ahead",
        "category": CATEGORY_WARNING,
        "meaning": "Traffic lights ahead, possibly following a high-speed rural stretch or obscured curve.",
        "instruction": "Be prepared to stop. Scan ahead for amber and red light transitions.",
        "penalty_risk": "Running red light leads to severe fines and points.",
        "shape": "Triangular with red, amber, and green signal lights",
        "priority": "Signal Advisory"
    },
    27: {
        "name": "Pedestrians Crossing",
        "category": CATEGORY_WARNING,
        "meaning": "Pedestrians frequently cross or walk along the roadway.",
        "instruction": "Yield unconditionally to pedestrians at marked or unmarked crosswalks. Never overtake waiting cars.",
        "penalty_risk": "Severe criminal and civil liability for pedestrian incidents.",
        "shape": "Triangular with walking pedestrian graphic",
        "priority": "Pedestrian Safety"
    },
    28: {
        "name": "Children Crossing / School Zone",
        "category": CATEGORY_WARNING,
        "meaning": "Proximity to a school, kindergarten, or playground with unpredictable child foot traffic.",
        "instruction": "Reduce speed below 30 km/h. Keep foot hovering over brake pedal; expect darting children.",
        "penalty_risk": "Maximum fine multiplier in school zones.",
        "shape": "Triangular with children running graphic",
        "priority": "High Vigilance"
    },
    29: {
        "name": "Bicycles Crossing",
        "category": CATEGORY_WARNING,
        "meaning": "Cyclists crossing or entering traffic from an intersecting bikeway.",
        "instruction": "Check mirrors and passenger blind spots before turning. Provide at least 1.5m clearance.",
        "penalty_risk": "Cyclist right-of-way violation penalty.",
        "shape": "Triangular with bicycle graphic",
        "priority": "Vulnerable Road User"
    },
    30: {
        "name": "Beware of Ice / Snow",
        "category": CATEGORY_WARNING,
        "meaning": "Road surface prone to black ice, frost, or sudden snow accumulation.",
        "instruction": "Drive cautiously at low speed. Test brakes gently on a straight section; avoid sudden moves.",
        "penalty_risk": "High weather accident risk.",
        "shape": "Triangular with snowflake graphic",
        "priority": "Winter Weather Hazard"
    },
    31: {
        "name": "Wild Animals Crossing",
        "category": CATEGORY_WARNING,
        "meaning": "Area with frequent crossing by deer, wild boar, or cattle, especially at dawn/dusk.",
        "instruction": "Use high beams when appropriate. If an animal appears, brake firmly; do not swerve into trees or oncoming lanes.",
        "penalty_risk": "Collision hazard.",
        "shape": "Triangular with leaping deer silhouette",
        "priority": "Wildlife Alert"
    },
    32: {
        "name": "End of All Speed & Passing Limits",
        "category": CATEGORY_SPEED_LIMIT,
        "meaning": "End of all previously posted restrictions including speed limits and passing bans.",
        "instruction": "Drive according to standard statutory highway rules and current weather/traffic density.",
        "penalty_risk": "Low.",
        "shape": "Circular white sign with multiple diagonal grey lines",
        "priority": "Restrictions Lifted"
    },
    33: {
        "name": "Turn Right Ahead",
        "category": CATEGORY_MANDATORY,
        "meaning": "Vehicles are required to execute a right turn ahead.",
        "instruction": "Signal right in advance, merge into right-turn lane, and turn right at the junction.",
        "penalty_risk": "Mandatory direction violation ticket.",
        "shape": "Blue circle with white right turn arrow",
        "priority": "Mandatory Direction"
    },
    34: {
        "name": "Turn Left Ahead",
        "category": CATEGORY_MANDATORY,
        "meaning": "Vehicles are required to execute a left turn ahead.",
        "instruction": "Signal left, yield to oncoming traffic if applicable, and turn left at the junction.",
        "penalty_risk": "Mandatory direction violation ticket.",
        "shape": "Blue circle with white left turn arrow",
        "priority": "Mandatory Direction"
    },
    35: {
        "name": "Ahead Only (Straight)",
        "category": CATEGORY_MANDATORY,
        "meaning": "Vehicles must proceed straight ahead only. Turning left or right is forbidden.",
        "instruction": "Remain in through lanes. Do not attempt left or right turns at this intersection.",
        "penalty_risk": "Illegal turn infraction.",
        "shape": "Blue circle with vertical upward white arrow",
        "priority": "Mandatory Direction"
    },
    36: {
        "name": "Go Straight or Turn Right",
        "category": CATEGORY_MANDATORY,
        "meaning": "Traffic may proceed straight through or make a right turn only.",
        "instruction": "Choose either straight or right lane. Left turn is prohibited.",
        "penalty_risk": "Illegal turn infraction.",
        "shape": "Blue circle with straight and right bifurcated arrows",
        "priority": "Mandatory Direction"
    },
    37: {
        "name": "Go Straight or Turn Left",
        "category": CATEGORY_MANDATORY,
        "meaning": "Traffic may proceed straight through or make a left turn only.",
        "instruction": "Choose either straight or left lane. Right turn is prohibited.",
        "penalty_risk": "Illegal turn infraction.",
        "shape": "Blue circle with straight and left bifurcated arrows",
        "priority": "Mandatory Direction"
    },
    38: {
        "name": "Keep Right",
        "category": CATEGORY_MANDATORY,
        "meaning": "Vehicles must pass the traffic island or obstacle on the right-hand side.",
        "instruction": "Steer to the right side of the median, splitter island, or road barrier.",
        "penalty_risk": "Wrong-way driving danger if disobeyed.",
        "shape": "Blue circle with downward diagonal right arrow",
        "priority": "Lane Guidance"
    },
    39: {
        "name": "Keep Left",
        "category": CATEGORY_MANDATORY,
        "meaning": "Vehicles must pass the traffic island or obstacle on the left-hand side.",
        "instruction": "Steer to the left side of the median divider or lane obstruction.",
        "penalty_risk": "Barrier collision / wrong-way hazard.",
        "shape": "Blue circle with downward diagonal left arrow",
        "priority": "Lane Guidance"
    },
    40: {
        "name": "Roundabout Mandatory",
        "category": CATEGORY_MANDATORY,
        "meaning": "Mandatory traffic circulation around roundabout in indicated direction.",
        "instruction": "Yield to vehicles already circulating inside the roundabout. Signal right when exiting.",
        "penalty_risk": "Failure to yield at roundabout junction fine.",
        "shape": "Blue circle with three circular arrows",
        "priority": "Roundabout Circulation"
    },
    41: {
        "name": "End of No Passing Zone",
        "category": CATEGORY_PROHIBITORY,
        "meaning": "The restriction banning overtaking of passenger vehicles has ended.",
        "instruction": "You may now overtake other vehicles when safe and sightlines are completely clear.",
        "penalty_risk": "Ensure oncoming lane is clear before passing.",
        "shape": "Circular white sign with slashed car silhouettes",
        "priority": "Passing Allowed"
    },
    42: {
        "name": "End of No Passing for Heavy Vehicles",
        "category": CATEGORY_PROHIBITORY,
        "meaning": "The restriction banning overtaking by heavy goods vehicles has ended.",
        "instruction": "Commercial trucks may pass slower vehicles where permitted by lane markings.",
        "penalty_risk": "Low.",
        "shape": "Circular white sign with slashed truck silhouette",
        "priority": "Restrictions Lifted"
    }
}

# Quick dictionary lookup by category
CATEGORIES_LIST = [
    CATEGORY_SPEED_LIMIT,
    CATEGORY_STOP_YIELD,
    CATEGORY_PROHIBITORY,
    CATEGORY_MANDATORY,
    CATEGORY_WARNING
]

def get_class_info(class_id: int) -> Dict[str, Any]:
    """Retrieve metadata dict for a class ID safely."""
    return TRAFFIC_SIGN_CLASSES.get(class_id, {
        "name": f"Unknown Sign #{class_id}",
        "category": "Unclassified",
        "meaning": "Traffic sign detected but specific classification pending.",
        "instruction": "Exercise caution and follow local road rules.",
        "penalty_risk": "Adhere to standard highway regulations.",
        "shape": "Unknown",
        "priority": "Advisory"
    })
