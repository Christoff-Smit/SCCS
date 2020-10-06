using Printf
using Dates
using PyCall
# using DelimitedFiles
# using Pkg
using DataFrames
# using Pandas
using CSV
# using Statistics
# using Plots
using Gadfly
using Colors
# using ScikitLearn
using WAV


StartingDate = Date("2020-09-01")
EndDate = Date("2020-11-30")

nrOfCredits = 24
recommendedHours = nrOfCredits*10

df = CSV.read("hour_log.txt",datarow=2)
println(df)
# println(DataFrames.describe(df))
totalHours = sum(df."Hours worked"[:])
# println(totalHours)

print("Do you want to add a new entry to the log? (y/n) ")
answer = readline()
if answer=="y"
    newEntry = 1
end
if answer=="n"
    newEntry = 0
end
# println(newEntry)
# print(typeof(newEntry))

if newEntry==1
    print("\nWELCOME TO THE HOUR LOGGING SCRIPT\n")
    print("Date: (YYYY-MM-DD format;\nt for today, yester for yesterday, tomor for tomorrow)\n")
    datestring = readline()
    # println(datestring)
    if datestring == "t"
        # datestring = Dates.format(Dates.today(),"yyyy-mm-dd")
        date = Dates.today()
        @printf("Date chosen as today (%s)\n",date)
    elseif datestring == "yester"
        date = Dates.today() - Dates.Day(1)
        @printf("Date chosen as yesterday (%s)\n",date)
    elseif datestring == "tomor"
        date = Dates.today() + Dates.Day(1)
        @printf("Date chosen as tomorrow (%s)\n",date)
    else
        date = Date(datestring,"y-m-d")
    end
    print("At what time did you start working (HH:MM)? ")
    start_time = DateTime(readline(), "HH:MM")
    println(start_time)
    print("How many hours of work did you complete? ")
    hours_worked = parse(Int,readline())
    println(hours_worked)
    stop_time = start_time+Dates.Hour(hours_worked)
    print("Give a short description of the work completed:\n")
    description = readline()
    println(df)
    entry_to_submit = [date,start_time,hours_worked,description,stop_time]
    println(entry_to_submit)
    push!(df,entry_to_submit)
end
# println(df)

stopTimes = Time[]
for row in eachrow(df)
    startTime = row[2]
    # println(startTime)
    hoursWorked = row[3]
    # println(hoursWorked)
    stopTime = startTime  + Dates.Hour(hoursWorked)
    # println(stopTime)
    push!(stopTimes,stopTime)
    # println(stopTimes)
end

# println(df)
df."Stop time" = stopTimes
sort!(df, (:Date,:"Start time"))
# println(df)

println("The following output is written to hour_log.txt")
println(df)

totalHours = sum(df."Hours worked"[:])
# println(totalHours)
percentage = totalHours/recommendedHours*100
@printf("You have completed %d/%d hours (%d%%) of the recommended work.\n",totalHours,recommendedHours,percentage)
idealHours = (Dates.today()-StartingDate)/(EndDate-StartingDate)*recommendedHours
idealPerc = idealHours/recommendedHours*100
@printf("Technically you should be at %d hours (%d%%)\n",idealHours,idealPerc)

CSV.write("hour_log.txt",df)



print("Do you want to update the backup file with these results? (y/n) ")
answer = readline()
if answer=="y"
    CSV.write("hour_log_backup.txt",df)
    println("Done!")
end














############## GANTT CHART ##############

nrOfEntries = size(df)[1]
yCount = [1:nrOfEntries][1]
# println(yCount)
# println(size(yCount))
# println(typeof(yCount))

textCutoffIndex = 25

# descriptions = Array{String,1}
descriptions = String[]
# println(descriptions)
# println(typeof(descriptions))
for description in df.Description
    if length(description)>textCutoffIndex
        push!(descriptions,string(description[1:textCutoffIndex],"..."))
    else
        push!(descriptions,description)
    end
    # println(descriptions)
end

gantt_chart_df = DataFrames.DataFrame(
    y = yCount,
    y_label=descriptions,
    start_time = df."Start time",
    stop_time = df."Stop time",
    id = yCount
)

y_label_dict = Dict(i=>gantt_chart_df[:y_label][i] for i in 1:nrOfEntries)

dates = Date[]
dates_plus_one = Date[]
for row in eachrow(df)
    push!(dates,row."Date")
    push!(dates_plus_one,row."Date"+Dates.Day(1))
end

gantt_chart_df.start_time = dates
gantt_chart_df.stop_time = dates_plus_one

# println(gantt_chart_df)

Gadfly.set_default_plot_size(18cm,14cm) #width,height

colormap = Colors.distinguishable_colors(nrOfEntries)

plot = Gadfly.plot(
    gantt_chart_df,
    Coord.cartesian(
        ymin=0,
        ymax=nrOfEntries*1,
        xmin=StartingDate,
        xmax=StartingDate+Dates.Day(35)#Dates.Day(nrOfEntries-1)
        ),
    layer(
        x=:start_time,
        xend=:stop_time,
        y=:y,
        yend=:y,
        # color=:id,
        # color=:y_label,
        color = rand(colormap,nrOfEntries),
        Geom.segment,
        Theme(line_width=4mm)
        ),
    Scale.y_continuous(labels=i->get(y_label_dict,i,"")),
    Guide.xlabel("Date"),
    Guide.ylabel("Entries"),
    Guide.title("Gantt chart of work completed"),
    Guide.yticks(ticks=1:nrOfEntries),
    # Guide.xticks(ticks=df."Date"),
    Theme(key_position=:none)
)