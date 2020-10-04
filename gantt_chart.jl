# workspace()

using Dates
using DataFrames
using Gadfly

y_labels = [
    # "MS 0: Oral presentation - Complex engineering problem",
    # "MS 1: Written Assignment - Complex engineering problem",
    "Review milestones 0 & 1",
    "Julia research & practice",
    "MS 2: Introduction and Literature Study",
    "MS 3: Specification Document",
    "MS 4: Oral Design Presentation",
    "MS 5: Design Document", #(Report on the design process)",
    "MS 6: Core Functionality Demo",
    # "MS 7: ?",
    # "MS 8: ?",
    # "MS 9: ?",
    "MS 10: Draft Report Submission",
    "MS 11: Final Report Submission",
    "MS 12: Submit (project brief)",
    "MS 13: Final Poster Submission",
    "MS 14: Final Oral Exam (Project Day)"
]
# println(y_labels)
#TODO shorten each y-axis label
ylabelCutoffIndex = 15


nrOfMilestones = length(y_labels)
# println(nrOfMilestones)

start_times = Date.(
    [
        # "2020-04-01",#MS 0
        # "2020-04-01",#MS 1
        "2020-09-01",#Review milestones 0 & 1
        "2020-09-04",#Julia research & practice
        "2020-09-07",#MS 2
        "2020-09-14",#MS 3
        "2020-09-18",#MS 4
        "2020-09-18",#MS 5
        "2020-10-01",#MS 6
        # "2020-04-01",#MS 7
        # "2020-04-01",#MS 8
        # "2020-04-01",#MS 9
        "2020-11-01",#MS 10
        "2020-11-9",#MS 11
        "2020-11-16",#MS 12
        "2020-11-20",#MS 13
        "2020-11-23"#MS 14
    ]
)
# println(start_times)

# StartingDate = start_times[1]
StartingDate = Date("2020-09-01")
# println(StartingDate)

stop_times = Date.(
    [
        # "2020-04-01",#MS 0
        # "2020-04-01",#MS 1
        "2020-09-04",#Review milestones 0 & 1
        "2020-09-07",#Julia research & practice
        "2020-09-14",#MS 2
        "2020-09-18",#MS 3
        "2020-10-01",#MS 4
        "2020-10-01",#MS 5
        "2020-11-01",#MS 6
        # "2020-04-01",#MS 7
        # "2020-04-01",#MS 8
        # "2020-04-01",#MS 9
        "2020-11-09",#MS 10
        "2020-11-16",#MS 11
        "2020-11-20",#MS 12
        "2020-11-23",#MS 13
        "2020-11-30"#MS 14
    ]
)
# println(stop_times)

# EndDate = stop_times[length(stop_times)]
EndDate = Date("2020-12-15")
# println(EndDate)

yCounter = [1:nrOfMilestones][1]
println(yCounter)

today = []
for i in yCounter
    push!(today,Dates.today())
end
# println(today)

df = DataFrames.DataFrame(
    y = yCounter,
    Milestones = y_labels,
    StartTimes = start_times,
    StopTimes = stop_times
    # id = yCounter
)
println(df)

y_labels_dict = Dict(i=>df[:Milestones][i] for i in 1:nrOfMilestones)
println(y_labels_dict)

#fixed_luminance:
l = 1
#croma:
c = 1
#starting point of hue variation:
h0 = 1
#amount of colors to generate:
n = nrOfMilestones
myColors = Gadfly.luv_rainbow(l,c,h0,n)


Gadfly.set_default_plot_size(18cm,12cm) #width,height
using Colors
colormap = Colors.distinguishable_colors(nrOfEntries)

plot = Gadfly.plot(
    df,
    Coord.cartesian(
        ymin=0,
        ymax=nrOfMilestones+5,
        xmin=StartingDate,
        xmax=EndDate
    ),
    layer(
        x=:StartTimes,
        xend=:StopTimes,
        y=:y,
        yend=:y,
        # color=:y, #TODO try using y as color column
        color = rand(colormap,nrOfEntries),
        Geom.segment,
        Theme(
            line_width=5mm
        )
    ),
    layer(
        x=today,
        # x=Dates.today(),
        y=yCounter,
        Geom.line,
        Theme(
            default_color="red",
            line_width = 0.5mm
        )
    ),
    Scale.y_continuous(labels=i->get(y_labels_dict,i,"")),
    Guide.xlabel("Dates"),
    Guide.ylabel("Milestones"),
    Guide.title("Project Gantt Chart"),
    Guide.yticks(ticks=1:nrOfMilestones),
    Guide.manual_discrete_key(
        "Legend",
        ["today"],
        color=["red"],
        shape=[Shape.hline]
    ),
    Theme(
        key_position=:none,
        plot_padding=[0mm,3mm,5mm,0mm],# [left, right, top, bottom]
    )
)