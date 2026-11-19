"""SimCity Israel look test v2 (rebuilt: the 14 Nov Blender script was lost with the local disk).
Noam 14 Nov: "wonderful! add music, and more stuff from 1900".
Toy plate of the whole land of Israel (West Bank and Gaza a slightly different shade, Green Line in green),
1900 start: Jaffa, Jerusalem Old City walls, Arab towns, first moshavot (red roofs), orange groves, Jaffa-Jerusalem
railway (1892), then Haifa-Tzemah railway (1905), Tel Aviv (1909), Degania (1910) ... to 2026.
Writes frames to OUT/f_####.png and OUT/proj.json (2D positions of labels per frame)."""
import bpy, bmesh, json, math, random, sys, os
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

GEO, OUT = sys.argv[sys.argv.index('--') + 1], sys.argv[sys.argv.index('--') + 2]
FR = int(sys.argv[sys.argv.index('--') + 3]) if len(sys.argv) > sys.argv.index('--') + 3 else 225
ONLY = sys.argv[sys.argv.index('--') + 4] if len(sys.argv) > sys.argv.index('--') + 4 else ''
os.makedirs(OUT, exist_ok=True)
random.seed(7)
g = json.load(open(GEO))
K = 10.0
def P(lon, lat): return Vector(((lon - 35.0) * K * math.cos(math.radians(31.5)), (lat - 31.5) * K, 0))

def year_of_frame(f):
    t = f / 15.0
    if t <= 4.0: return 1900 + t / 4.0 * 10.0                 # 1900 to 1910 slowly: the 1900 stuff
    return 1910 + (t - 4.0) / 11.0 * 116.0                     # 1910 to 2026
def frame_of_year(y):
    if y <= 1910: return (y - 1900) / 10.0 * 4.0 * 15
    return (4.0 + (y - 1910) / 116.0 * 11.0) * 15

# ---------- scene ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE' if 'BLENDER_EEVEE' in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items] else 'BLENDER_EEVEE_NEXT'
sc.render.resolution_x, sc.render.resolution_y = 540, 960
sc.render.fps = 15
try: sc.eevee.taa_render_samples = 8
except Exception: pass
sc.render.film_transparent = False
world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.012, 0.03, 0.09, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 0.9
sc.view_settings.view_transform = 'Standard'

def mat(name, rgb, rough=0.8, emit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*rgb, 1); b.inputs['Roughness'].default_value = rough
    if emit:
        b.inputs['Emission Color'].default_value = (*rgb, 1); b.inputs['Emission Strength'].default_value = emit
    return m
def srgb(h):
    h = h.lstrip('#'); c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple((x / 12.92) if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)
M_SAND = mat('sand', srgb('D9C08C')); M_TERR = mat('terr', srgb('C4A673')); M_NB = mat('nb', srgb('A8946A'))
M_SEA = mat('sea', srgb('163261'), 0.35); M_GL = mat('gl', srgb('2ECC71'), 0.5, 2.0)
M_STONE = mat('stone', srgb('E8DCC4')); M_WHITE = mat('white', srgb('F4F6FA')); M_GLASS = mat('glass', srgb('7FB8E6'), 0.25)
M_ROOF = mat('roof', srgb('C2452D')); M_GOLD = mat('gold', srgb('F2C230'), 0.3, 0.4); M_RAIL = mat('rail', srgb('3A3A44'))
M_ROAD = mat('road', srgb('4A4E5A')); M_FIELD = mat('field', srgb('5FAE4E')); M_GROVE = mat('grove', srgb('2F7D3A'))
M_WALL = mat('wall', srgb('CDB98F'))

def poly_obj(name, rings, m, h=0.25, z=0.0, step=1):
    bm = bmesh.new()
    for ring in rings:
        pts = ring[::step]
        if len(pts) < 3: continue
        vs = [bm.verts.new(P(*p[:2]) + Vector((0, 0, z))) for p in pts]
        try: f = bm.faces.new(vs)
        except ValueError: continue
        r = bmesh.ops.extrude_face_region(bm, geom=[f])
        bmesh.ops.translate(bm, vec=Vector((0, 0, h)), verts=[e for e in r['geom'] if isinstance(e, bmesh.types.BMVert)])
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = False
    o = bpy.data.objects.new(name, me); o.data.materials.append(m); sc.collection.objects.link(o); return o
def rings_of(gm):
    cs = gm['coordinates'] if gm['type'] == 'MultiPolygon' else [gm['coordinates']]
    return [c[0] for c in cs]

# sea plane
bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, -0.05)); bpy.context.object.data.materials.append(M_SEA)
# neighbours (muted), then the land of Israel on top
for code in ('LBN', 'SYR', 'JOR'):
    poly_obj('nb_' + code, rings_of(g['local'][code]), M_NB, 0.18, 0, 1)
poly_obj('nb_sinai', rings_of(g['sinai']), M_NB, 0.18, 0, 1)
poly_obj('isr', rings_of(g['isr49']), M_SAND, 0.25, 0.0)
poly_obj('golan', rings_of(g['golan']), M_SAND, 0.25, 0.0)
poly_obj('wb', rings_of(g['wb']), M_TERR, 0.25, 0.001)
poly_obj('gaza', rings_of(g['gaza']), M_TERR, 0.25, 0.001)

def tube(name, pts, m, r=0.03, z=0.27, closed=False):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = r; cu.bevel_resolution = 1
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        v = P(*p[:2]); sp.points[i].co = (v.x, v.y, z, 1)
    sp.use_cyclic_u = closed
    o = bpy.data.objects.new(name, cu); o.data.materials.append(m); sc.collection.objects.link(o); return o
for nm in ('wb', 'gaza'):
    for i, ring in enumerate(rings_of(g[nm])): tube('gl_%s%d' % (nm, i), ring[::1], M_GL, 0.028, 0.27, True)

def grow(o, y0, dur_years=3.0, final=None):
    """object pops up from scale 0 at year y0"""
    f0 = frame_of_year(y0); f1 = frame_of_year(y0 + dur_years)
    fs = final or tuple(o.scale)
    o.scale = (0, 0, 0.0001) if y0 > 1900 else fs; o.keyframe_insert('scale', frame=max(0, f0 - 1))
    o.scale = fs; o.keyframe_insert('scale', frame=max(1, f1) if y0 > 1900 else 0)

# shared block meshes
def boxmesh(name):
    bpy.ops.mesh.primitive_cube_add(size=1); o = bpy.context.object; me = o.data; me.name = name
    bpy.data.objects.remove(o); return me
BOX = boxmesh('box')
def roofmesh():
    bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=0.72, radius2=0.0, depth=0.5)
    me = bpy.data.meshes.new('roof'); bm.to_mesh(me); bm.free(); return me
ROOF = roofmesh()
def domemesh():
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=0.5)
    me = bpy.data.meshes.new('dome'); bm.to_mesh(me); bm.free(); return me
DOME = domemesh()
BOX.materials.append(M_WHITE); ROOF.materials.append(M_ROOF); DOME.materials.append(M_GOLD)

LABELS = []      # (text, sub, world pos, frame show from, frame show to)
def label(text, sub, lon, lat, y0, hold=2.2, z=0.8):
    f0 = frame_of_year(y0) if y0 > 1900 else 3
    LABELS.append((text, sub, tuple(P(lon, lat) + Vector((0, 0, z))), f0, f0 + hold * 15))

def block(x, y, w, h, m, y_app, roof=False):
    o = bpy.data.objects.new('b', BOX)
    o.location = (x, y, 0.25 + h / 2); o.scale = (w, w, h)
    sc.collection.objects.link(o)
    o.material_slots[0].link = 'OBJECT'; o.material_slots[0].material = m
    grow_z(o, y_app, h, 0.25)
    if roof:
        r = bpy.data.objects.new('r', ROOF); r.location = (x, y, 0.25 + h + 0.12); r.scale = (w * 0.9, w * 0.9, w * 0.6)
        r.rotation_euler = (0, 0, math.pi / 4); sc.collection.objects.link(r)
        if r.material_slots: r.material_slots[0].link = 'OBJECT'; r.material_slots[0].material = M_ROOF
        else: r.data.materials.append(M_ROOF)
        grow(r, y_app + 0.5, 1.5)
    return o
def grow_z(o, y_app, h, base):
    f1 = frame_of_year(y_app); f0 = f1 - 6
    if y_app <= 1900:
        return
    sx, sy = o.scale.x, o.scale.y
    o.scale = (0, 0, 0.0001); o.location.z = base; o.keyframe_insert('scale', frame=max(0, f0)); o.keyframe_insert('location', frame=max(0, f0))
    o.scale = (sx, sy, h); o.location.z = base + h / 2; o.keyframe_insert('scale', frame=f1); o.keyframe_insert('location', frame=f1)

def town(lon, lat, y_from, y_to, n_from, n_to, radius, m, hmin=0.12, hmax=0.35, wmin=0.08, wmax=0.16, roof=False, tall_after=None):
    c = P(lon, lat); n = n_to
    for k in range(n):
        y_app = y_from if k < n_from else y_from + (y_to - y_from) * ((k - n_from) / max(1, n - n_from)) ** 0.8
        rr = radius * math.sqrt((k + 0.5) / n) * (0.9 + 0.2 * random.random()); th = k * 2.39996 + random.random() * 0.3
        h = random.uniform(hmin, hmax)
        mm = m
        if tall_after and y_app >= tall_after[0]: h = random.uniform(hmin, hmax * tall_after[1]); mm = random.choice([M_WHITE, M_GLASS, M_WHITE])
        block(c.x + rr * math.cos(th), c.y + rr * math.sin(th), random.uniform(wmin, wmax), h, mm, y_app, roof)

# ---------- 1900: already there ----------
# Jaffa: old port town, dense stone; Jerusalem Old City walls + Dome; Arab towns; Haifa small
town(34.752, 32.053, 1900, 1948, 70, 110, 0.28, M_STONE, 0.06, 0.16, 0.05, 0.09)
label('Jaffa', 'port town, 1900', 34.752, 32.053, 1900)
oc = P(35.2316, 31.7767)
pts = [(oc.x - 0.32, oc.y - 0.3), (oc.x + 0.32, oc.y - 0.3), (oc.x + 0.32, oc.y + 0.3), (oc.x - 0.32, oc.y + 0.3)]
for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
    w = bpy.data.objects.new('wall', BOX); sc.collection.objects.link(w)
    w.location = ((ax + bx) / 2, (ay + by) / 2, 0.25 + 0.08); w.scale = (abs(bx - ax) + 0.05, abs(by - ay) + 0.05, 0.16)
    if w.material_slots: w.material_slots[0].link = 'OBJECT'; w.material_slots[0].material = M_WALL
for k in range(55):
    block(oc.x + random.uniform(-0.27, 0.27), oc.y + random.uniform(-0.25, 0.25), random.uniform(0.05, 0.08), random.uniform(0.05, 0.12), M_STONE, 1900)
d = bpy.data.objects.new('dome', DOME); sc.collection.objects.link(d); d.location = (oc.x + 0.08, oc.y + 0.05, 0.42); d.scale = (0.16, 0.16, 0.14)
if d.material_slots: d.material_slots[0].link = 'OBJECT'; d.material_slots[0].material = M_GOLD
label('Jerusalem', 'Old City walls, 1900', 35.2316, 31.7767, 1900, 3.0)
town(35.21, 31.79, 1900, 2026, 0, 260, 0.75, M_STONE, 0.1, 0.4, 0.07, 0.14, tall_after=(1990, 1.6))   # new city outside the walls
for nm, lon, lat, n in [('Hebron', 35.10, 31.53, 30), ('Nazareth', 35.30, 32.70, 26), ('Gaza', 34.46, 31.50, 34), ('Acre', 35.07, 32.92, 24),
                        ('Safed', 35.50, 32.96, 18), ('Tiberias', 35.53, 32.79, 14), ('Nablus', 35.26, 32.22, 26)]:
    town(lon, lat, 1900, 2026, n, n * 3, 0.22 + n * 0.004, M_STONE, 0.06, 0.18, 0.05, 0.1)
town(34.99, 32.81, 1900, 2026, 10, 190, 0.6, M_STONE, 0.08, 0.35, 0.06, 0.13, tall_after=(1960, 1.8))
label('Haifa', 'small port, 1900', 34.99, 32.81, 1900)
# first moshavot: red roofs among orange groves
for nm, lon, lat, yr in [('Petah Tikva', 34.887, 32.087, 1878), ('Rishon LeZion', 34.80, 31.964, 1882), ('Zikhron Ya\'akov', 34.95, 32.57, 1882),
                         ('Rosh Pinna', 35.54, 32.97, 1882), ('Rehovot', 34.81, 31.894, 1890), ('Hadera', 34.92, 32.44, 1891)]:
    town(lon, lat, 1900, 2026, 9, 120 if nm in ('Petah Tikva', 'Rishon LeZion') else 70, 0.4, M_WHITE, 0.06, 0.12, 0.06, 0.09, roof=False, tall_after=(1975, 2.2))
    c = P(lon, lat)
    for k in range(9):
        th = k * 2.39996; rr = 0.12 * math.sqrt(k + 0.5)
        r = bpy.data.objects.new('r', ROOF); r.location = (c.x + rr * math.cos(th), c.y + rr * math.sin(th), 0.25 + 0.12); r.scale = (0.07, 0.07, 0.07)
        r.rotation_euler = (0, 0, math.pi / 4); sc.collection.objects.link(r)
        if r.material_slots: r.material_slots[0].link = 'OBJECT'; r.material_slots[0].material = M_ROOF
    label(nm, 'founded %d' % yr, lon, lat, 1900 + (0.4 if nm == 'Petah Tikva' else 0.9 if nm == 'Rishon LeZion' else 1.6), 2.0)
# orange groves around Jaffa
for k in range(140):
    c = P(34.80 + random.uniform(-0.04, 0.09), 32.04 + random.uniform(-0.09, 0.08))
    if (c - P(34.752, 32.053)).length < 0.3: continue
    o = bpy.data.objects.new('gv', DOME); sc.collection.objects.link(o); o.location = (c.x, c.y, 0.29); o.scale = (0.05, 0.05, 0.05)
    if o.material_slots: o.material_slots[0].link = 'OBJECT'; o.material_slots[0].material = M_GROVE
label('Orange groves', 'Jaffa oranges, 1900', 34.86, 32.02, 1902.2, 1.8)
# railways
jj = [(34.77, 32.05), (34.86, 31.93), (34.90, 31.82), (34.98, 31.77), (35.08, 31.74), (35.17, 31.75), (35.22, 31.76)]
rj = tube('rail_jj', jj, M_RAIL, 0.035, 0.28)
label('Jaffa to Jerusalem railway', 'opened 1892', 34.98, 31.77, 1903.0, 2.0)
hz = [(35.00, 32.80), (35.10, 32.74), (35.25, 32.68), (35.40, 32.68), (35.55, 32.70), (35.62, 32.71)]
rh = tube('rail_hz', hz, M_RAIL, 0.035, 0.28); rh.data.bevel_factor_end = 0.0
rh.data.keyframe_insert('bevel_factor_end', frame=frame_of_year(1904.6)); rh.data.bevel_factor_end = 1.0; rh.data.keyframe_insert('bevel_factor_end', frame=frame_of_year(1905.6))
label('Haifa to Tzemah railway', 'opened 1905', 35.30, 32.68, 1905.0, 1.8)

# ---------- 1909 on ----------
town(34.78, 32.08, 1909, 2026, 0, 420, 1.0, M_WHITE, 0.08, 0.35, 0.07, 0.14, tall_after=(1985, 3.5))
label('Tel Aviv', 'founded 1909', 34.78, 32.08, 1909.2, 2.2)
for nm, lon, lat, yr, n, rad in [('Degania', 35.57, 32.70, 1910, 10, 0.12), ('Nahalal', 35.20, 32.69, 1921, 14, 0.15), ('Netanya', 34.86, 32.33, 1929, 120, 0.45),
                                 ('Beersheba', 34.79, 31.25, 1950, 170, 0.6), ('Ashkelon', 34.57, 31.67, 1950, 110, 0.45), ('Eilat', 34.95, 29.56, 1951, 70, 0.35),
                                 ('Dimona', 35.03, 31.07, 1955, 50, 0.3), ('Ashdod', 34.65, 31.80, 1956, 150, 0.55), ('Modiin', 35.01, 31.90, 1996, 90, 0.4)]:
    town(lon, lat, yr, 2026, 0, n, rad, M_WHITE, 0.07, 0.3, 0.07, 0.13, tall_after=(1990, 2.0))
    label(nm, 'founded %d' % yr if yr < 1948 or nm in ('Eilat', 'Dimona', 'Ashdod', 'Modiin') else 'grows from %d' % yr, lon, lat, yr + 0.3, 1.6)
# fields spread (kibbutz and moshav farmland)
for k in range(200):
    lon, lat, yr = random.choice([(35.25, 32.62, 1922), (35.45, 32.55, 1935), (34.70, 31.45, 1950), (34.62, 31.30, 1955), (35.55, 32.75, 1912), (35.0, 32.45, 1935)])
    c = P(lon + random.uniform(-0.12, 0.12), lat + random.uniform(-0.08, 0.08)); w = random.uniform(0.08, 0.16)
    o = bpy.data.objects.new('fd', BOX); sc.collection.objects.link(o); o.location = (c.x, c.y, 0.255); o.scale = (w, w * random.uniform(0.6, 1.4), 0.01)
    if o.material_slots: o.material_slots[0].link = 'OBJECT'; o.material_slots[0].material = M_FIELD
    grow(o, yr + random.uniform(0, 25), 2.0)
# roads
for pts, yr in [([(34.80, 32.07), (34.86, 32.33), (34.92, 32.44), (34.99, 32.79)], 1950), ([(34.80, 32.07), (34.95, 31.85), (35.20, 31.79)], 1952),
                ([(34.80, 32.05), (34.79, 31.25), (34.95, 29.56)], 1958), ([(34.95, 32.6), (34.98, 32.1), (34.88, 31.6), (34.85, 31.3)], 2002)]:
    r = tube('road', pts, M_ROAD, 0.03, 0.27); r.data.bevel_factor_end = 0.0
    r.data.keyframe_insert('bevel_factor_end', frame=frame_of_year(yr)); r.data.bevel_factor_end = 1.0; r.data.keyframe_insert('bevel_factor_end', frame=frame_of_year(yr + 6))

# ---------- light + camera ----------
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.2; sun.angle = 0.2
so = bpy.data.objects.new('sun', sun); so.rotation_euler = (math.radians(50), math.radians(10), math.radians(35)); sc.collection.objects.link(so)
cam = bpy.data.cameras.new('cam'); cam.lens = 32; co = bpy.data.objects.new('cam', cam); sc.collection.objects.link(co); sc.camera = co
tgt = bpy.data.objects.new('tgt', None); sc.collection.objects.link(tgt)
c = co.constraints.new('TRACK_TO'); c.target = tgt; c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'
keys = [  # (second, camera lon, lat, height, target lon, lat)
    (0.0, 34.55, 31.80, 2.4, 34.85, 32.02),     # low over the sea, looking at Jaffa, groves, moshavot
    (2.4, 34.80, 31.45, 3.6, 35.12, 31.80),     # along the railway to Jerusalem
    (4.2, 34.70, 31.70, 3.8, 34.95, 32.10),
    (6.5, 34.60, 31.55, 5.5, 34.90, 32.10),     # Tel Aviv grows
    (10.0, 34.30, 30.60, 11.0, 34.95, 31.60),
    (15.0, 34.20, 29.70, 19.0, 35.00, 31.30)]   # whole country by 2026
for s, lo, la, h, tlo, tla in keys:
    f = int(round(s * 15)); co.location = P(lo, la) + Vector((0, 0, h)); co.keyframe_insert('location', frame=f)
    tgt.location = P(tlo, tla); tgt.keyframe_insert('location', frame=f)
for ob in (co, tgt):
    if ob.animation_data and ob.animation_data.action:
        try:
            for fc in ob.animation_data.action.fcurves:
                for kp in fc.keyframe_points: kp.interpolation = 'BEZIER'; kp.easing = 'AUTO'
        except Exception: pass
sc.frame_start, sc.frame_end = 0, FR - 1

# ---------- render ----------
proj = {}
frames = range(0, FR) if not ONLY else [int(x) for x in ONLY.split(',')]
for f in frames:
    sc.frame_set(f)
    sc.render.filepath = os.path.join(OUT, 'f_%04d.png' % f)
    bpy.ops.render.render(write_still=True)
    pl = []
    for text, sub, pos, a, b in LABELS:
        if a - 2 <= f <= b + 2:
            v = world_to_camera_view(sc, co, Vector(pos))
            if 0 < v.x < 1 and 0 < v.y < 1 and v.z > 0: pl.append([text, sub, v.x, 1 - v.y, a, b])
    proj[f] = dict(year=year_of_frame(f), labels=pl)
    json.dump(proj, open(os.path.join(OUT, 'proj%s.json' % ('_test' if ONLY else '')), 'w'))
print('done')
