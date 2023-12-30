extract query for top five projects full

select jira_issue_report.id, jira_issue_report.project_name, jira_issue_report.title, jira_issue_report.description, jira_issue_report.priority
from jira_issue_report
where project_name in ('RichFaces', 'Tools (JBoss Tools)', 'Grails', 'Hadoop Common', 'HBase') and type='Bug' 
and id not in (select issue_report_id  as id 
			   from jira_issue_changelog_item where field_name='priority' and original_value='Major')



extract query for top five projects log 

select report.project_name , report.title, report.description, logg.new_value as new_priority
from jira_issue_changelog_item as logg, jira_issue_report as report
where report.id=logg.issue_report_id and report.project_name in ('RichFaces', 'Tools (JBoss Tools)', 'Grails', 'Hadoop Common', 'HBase')
and report.type='Bug' and logg.field_name='priority' and logg.original_value='Major'
