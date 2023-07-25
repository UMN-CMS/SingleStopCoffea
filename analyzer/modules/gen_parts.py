import itertools as it
from analyzer.core import analyzerModule, ModuleType
import awkward as ak

def isGoodGenParticle(particle):
    return particle.hasFlags("isLastCopy", "fromHardProcess") & ~(
        particle.hasFlags("fromHardProcessBeforeFSR")
        & ((abs(particle.pdgId) == 1) | (abs(particle.pdgId) == 3))
    )

def createGoodChildren(gen_particles, children):
    # x = ak.singletons(gen_particles.children)
    good_child_mask = isGoodGenParticle(children)
    while not ak.all(ak.ravel(good_child_mask)):
        good_child_mask = isGoodGenParticle(children)
        maybe_good_children = children[~good_child_mask].children
        maybe_good_children = ak.flatten(maybe_good_children, axis=3)
        children = ak.concatenate(
            [children[good_child_mask], maybe_good_children], axis=2
        )
    return children

@analyzerModule("good_gen", ModuleType.MainProducer, require_tags=["signal"])
def goodGenParticles(events):
    test = createGoodChildren(events.GenPart, events.GenPart.children)

    def get(x):
        return [y.pdgId for y in x[0]]

    events["GenPart", "good_children"] = test
    gg = events.GenPart[isGoodGenParticle(events.GenPart)]
    bs = gg.good_children[abs(gg.good_children.pdgId) == 5]
    stop = ak.all(abs(gg[:, 0].pdgId) == 1000006)
    bad = ak.all(abs(gg[:, 1].pdgId) == 1000024)
    x = gg[abs(gg.pdgId) == 1000024][:, 0]
    t = gg[abs(gg.pdgId) == 1000006][:, 0]
    x_children = gg.good_children[abs(gg.pdgId) == 1000024]
    s_children = gg.good_children[abs(gg.pdgId) == 1000006]
    xb = ak.flatten(ak.flatten(x_children[abs(x_children.pdgId) == 5]))
    xd = ak.flatten(ak.flatten(x_children[abs(x_children.pdgId) == 1]))
    xs = ak.flatten(ak.flatten(x_children[abs(x_children.pdgId) == 3]))
    sb = ak.flatten(ak.flatten(s_children[abs(s_children.pdgId) == 5]))
    events["SignalParticles"] = ak.zip(
        dict(
            stop=t,
            chi=x,
            stop_b=sb,
            chi_b=xb,
            chi_d=xd,
            chi_s=xs,
        )
    )
    events["SignalQuarks"] = ak.concatenate(
        [ak.singletons(val) for val in [sb, xb, xd, xs]], axis=1
    )
    return events
